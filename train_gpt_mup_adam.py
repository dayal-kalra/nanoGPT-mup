"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import re
from contextlib import nullcontext
import argparse

import numpy as np
import pandas as pd

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.gpt_mup import GPTConfig, GPT
from utils.schedules_utils import warmup_stable_decay

torch.backends.cuda.matmul.allow_tf32 = True # enable tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # enable tf32 on cudnn

def get_batch(cfg, split, device):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    # NOTE: use uint16 reduced the dataset size during read; ONLY USED WHEN the vocab size is less than 2^16
    if split == 'train':
        data = np.memmap(os.path.join(cfg.data_dir, 'train.bin'), dtype = np.uint16, mode = 'r')
    else:
        data = np.memmap(os.path.join(cfg.data_dir, 'val.bin'), dtype = np.uint16, mode = 'r')

    # select batch_size (valid) starting points
    ix = torch.randint(len(data) - cfg.context_len, (cfg.batch_size,))
    # extract inputs
    x = torch.stack([torch.from_numpy((data[i:i+cfg.context_len]).astype(np.int64)) for i in ix])
    # extract outputs
    y = torch.stack([torch.from_numpy((data[i+1:i+1+cfg.context_len]).astype(np.int64)) for i in ix])

    if cfg.device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking = True), y.pin_memory().to(device, non_blocking = True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(cfg, model, ctx, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_steps)
        for k in range(cfg.eval_steps):
            X, Y = get_batch(cfg, split, device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def create_train_state(cfg, device):

    # create model
    gpt_conf = GPTConfig(
        context_len = cfg.context_len,
        vocab_size = cfg.vocab_size,
        num_layers = cfg.num_layers,
        num_heads = cfg.num_heads,
        embd_dim = cfg.embd_dim,
        bias = cfg.use_bias,
        init_var = cfg.init_var,
    )
    
    model = GPT(gpt_conf)
    model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            mean = param.mean().item()
            var = param.var().item()
            if cfg.verbose:
                if 'scale' in name or 'bias' in name:
                    print(f'Parameter: {name}, mean: {mean:.4f}, var: {var:.4f}')
                else:
                    print(f'Parameter: {name}, mean: {mean:.4f}, var: {var:.4f}, 1/var: {1.0/var:0.4f}')
                        
    # create optimizer
    optim = model.muAdamW(
        learning_rate = cfg.lr_init, 
        betas = (cfg.beta1, cfg.beta2), 
        weight_decay = cfg.weight_decay, 
        eps = cfg.epsilon,
        device_type = cfg.device_type
    ) 
    
    return model, optim

def train_and_evaluate(cfg, device):
    
    model, optim = create_train_state(cfg, device)

    num_params, embd_params = model.get_num_params()
    print("number of parameters: %.2fM" % (num_params/1e6,))
    print("number of embd params: %.2fM" % (embd_params/1e6,))

    # compile the model
    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # wrap model into DDP container
    if cfg.ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    train_results = list()
    eval_results = list()

    lr_step = cfg.lr_init

    ctx = nullcontext() if cfg.device_type == 'cpu' else torch.amp.autocast(device_type = cfg.device_type, dtype = cfg.ptdtype)
    # initialize a GradScaler. Only enable it for float16 computations
    scaler = torch.amp.GradScaler('cuda', enabled = (cfg.dtype == 'float16'))

    # training loop
    X, Y = get_batch(cfg, 'train', device) # fetch the very first batch


    for step in range(cfg.num_steps+1):

        ### EVALUATE ##
        if step % cfg.eval_interval == 0 and cfg.master_process:
            losses = estimate_loss(cfg, model, ctx, device)
            result = np.array([step, lr_step, losses['train'], losses['val']])
            eval_results.append(result)

            # Save to the data after every evaluation
            df_eval = pd.DataFrame(eval_results, columns = ['step', 'lr_step', 'train_loss', 'val_loss'])
            df_eval['num_params'] = num_params
            df_eval['embd_params'] = embd_params
            df_eval.to_csv(cfg.evals_path, index = False)

        ### TRAIN STEP ###
        cosine_step = step - cfg.warmup_steps
        # get the learning rate
        lr_step = warmup_stable_decay(
            step = step,
            init_value = cfg.lr_init,
            peak_value = cfg.lr_peak,
            min_value = cfg.lr_min,
            num_steps = cfg.num_steps,
            warmup_steps = cfg.warmup_steps,
            stable_steps = cfg.stable_steps,
            warmup_exponent = cfg.warmup_exponent,
            decay_schedule_name = cfg.decay_schedule_name,
            decay_exponent = cfg.decay_exponent
        )
        
        # update the learning rate
        for param_group in optim.param_groups:
            param_group['lr'] = lr_step * param_group.get('lr_scale', 1.0)

        for micro_step in range(cfg.gradient_accumulation_steps):
            if cfg.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                model.require_backward_grad_sync = (micro_step == cfg.gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg.gradient_accumulation_steps # scale the loss to account for gradient accumulation
            
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(cfg, 'train', device)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        
        # clip the gradient
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        # step the optimizer and scaler if training in fp16
        # 1. takes a step if no overflows encountered
        # 2. skips the training step otherwise
        scaler.step(optim)

        # adjusts the scaling factor for next iteration
        scaler.update() 

        # flush the gradients as soon as we can, no need for this memory anymore
        optim.zero_grad(set_to_none = True)

        if step % cfg.log_interval == 0 and cfg.master_process:
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * cfg.gradient_accumulation_steps
            print(f"step: {step}, lr: {lr_step:0.1e}, loss: {lossf:.4f}")
            result = np.array([step, lr_step, lossf])
            train_results.append(result)

    # training data results
    train_results = np.array(train_results)
    df_train = pd.DataFrame(train_results, columns = ['step', 'lr_step', 'loss_step'])
    df_train['num_params'] = num_params
    df_train['embd_params'] = embd_params
    df_train.to_csv(cfg.train_path, index = False)

    # destroy the DDP process group
    if cfg.ddp:
        destroy_process_group()

    return 

### CONFIG ###

parser = argparse.ArgumentParser(description = 'Experiment parameters')
parser.add_argument('--cluster', type = str, default = 'zaratan')

### dataset 
parser.add_argument('--dataset_name', type = str, default = 'fineweb')
parser.add_argument('--vocab_size', type = int, default = 50304)
# GPU + DDP

### model
parser.add_argument('--init_var', type = float, default = 1.0)
parser.add_argument('--num_layers', type = int, default = 12)
parser.add_argument('--num_heads', type = int, default = 12)
parser.add_argument('--head_dim', type = int, default = 64)
parser.add_argument('--embd_dim', type = int, default = 768)
parser.add_argument('--bias', type = str, default = 'False')
parser.add_argument('--context_len', type = int, default = 1024)
parser.add_argument('--compile', type = bool, default = True)

### optimization
# adamw optimizer
parser.add_argument('--optim_name', type = str, default = 'AdamW')
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--lr_peak', type = float, default = 1e-03)
parser.add_argument('--lr_min_factor', type = lambda x: float('inf') if x.lower() == 'inf' else float(x), default = float('inf'))
parser.add_argument('--beta1', type = float, default = 0.9)
parser.add_argument('--beta2', type = float, default = 0.95)
parser.add_argument('--epsilon', type = float, default = 1e-08)
parser.add_argument('--weight_decay', type = float, default = 0.1)
# training time; batch_size; and other things
parser.add_argument('--num_steps', type = int, default = 2_000)
parser.add_argument('--warmup_steps', type = int, default = 500)
parser.add_argument('--warmup_exponent', type = float, default = 1.0)
parser.add_argument('--stable_steps', type = int, default = 0)
parser.add_argument('--decay_schedule_name', type = str, default = 'cosine') # 'cosine' or 'polynomial'
parser.add_argument('--decay_exponent', type = float, default = 1.0)
parser.add_argument('--gradient_accumulation_steps', type = int, default = 40)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--grad_clip', type = float, default = 1.0)
### evaluation
parser.add_argument('--log_interval', type = int, default = 1)
parser.add_argument('--eval_interval', type = int, default = 100)
parser.add_argument('--eval_steps', type = int, default = 200)
parser.add_argument('--verbose', type = bool, default = False)


cfg = parser.parse_args()

cfg.use_bias = True if cfg.bias == 'True' else False
cfg.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
cfg.embd_dim = cfg.head_dim * cfg.num_heads
cfg.lr_min = cfg.lr_peak / cfg.lr_min_factor

# DDP settings
device = 'cuda'
backend = 'nccl'
cfg.ddp = int(os.environ.get('RANK', -1)) != -1 # check if this is a DDP run

if cfg.ddp:
    init_process_group(backend = backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    cfg.master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert cfg.gradient_accumulation_steps % ddp_world_size == 0
    cfg.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    cfg.master_process = True
    seed_offset = 0
    ddp_world_size = 1

torch.manual_seed(1337 + seed_offset)

cfg.device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
cfg.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]

### Data loader ###
cfg.data_dir = os.path.join('data', cfg.dataset_name)
print(f"data_dir: {cfg.data_dir}")

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(cfg.data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Save results
train = True

if cfg.master_process:
    os.makedirs('results', exist_ok=True)

# Create a descriptive name based on the model and training parameters
base_filename = (
    f'{cfg.dataset_name}_'
    f'v{cfg.vocab_size}_'
    f'gpt_mup_'
    f'var{cfg.init_var:0.1f}_'
    f'd{cfg.num_layers}_'
    f'h{cfg.num_heads}_'
    f'n{cfg.embd_dim}_'
    f'c{cfg.context_len}_'
    f'{cfg.optim_name}_'
    f'Tw{cfg.warmup_steps}_'
    f'r{cfg.warmup_exponent}_'
    f'Ts{cfg.stable_steps}_'
    f'p{cfg.decay_exponent}_'
    f'T{cfg.num_steps}_'
    f'ga{cfg.gradient_accumulation_steps}_'
    f'lr{cfg.lr_peak:.0e}_'
    f'lr{cfg.lr_min_factor}_'
    f'wd{cfg.weight_decay}_'
    f'bs{cfg.batch_size}_'
    f'b{cfg.beta1}_{cfg.beta2}_'
    f'eps{cfg.epsilon}_'
    f'gc{cfg.grad_clip}'
)

train_filename = f'train_{base_filename}.csv'
cfg.train_path = os.path.join('results', train_filename)

evals_filename = f'eval_{base_filename}.csv'
cfg.evals_path = os.path.join('results', evals_filename)

print(train_filename)
if not os.path.exists(cfg.train_path):
    train_and_evaluate(cfg, device)

