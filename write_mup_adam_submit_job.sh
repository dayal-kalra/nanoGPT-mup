echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:59:00
#SBATCH --job-name=gpt-$1
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
#SBATCH --mem=64G
#SBATCH --constraint=volta32gb
#SBATCH --gres=gpu:1
###SBATCH --gpus=h100:1
#SBATCH --partition=learnlab

# Environment setup
source /private/home/dayal/miniforge3/bin/activate
conda activate torch

cd \$SLURM_SUBMIT_DIR

time srun python3 train_gpt_mup_adam.py \
    --dataset_name fineweb \
    --num_layers 12 \
    --num_heads 12 \
    --init_var 1.0 \
    --batch_size 32 \
    --gradient_accumulation_steps 20 \
    --lr_peak $1 \
    --lr_min_factor inf \
    --weight_decay 0.0 \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup_steps 100 \
    --num_steps 1_000 \
    --eval_interval 200" > submit_job.sh
