#!/bin/bash
#SBATCH --job-name=papez
#SBATCH --output=train_output.out
#SBATCH --error=job_error.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research   
#SBATCH --time=1440 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate Conda
source /home/joberant/NLP_2425a/doronaloni/anaconda3/etc/profile.d/conda.sh
conda activate audio_env

# Debug CUDA status
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
python -c "import torch; torch.cuda.is_available(); print('CUDA check passed')"

# For training:
# python train.py

# For evaluation:
# python test.py