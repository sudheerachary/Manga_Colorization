#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END

module add opencv
module add cuda/8.0
module add cudnn/6-cuda-8.0

python GAN_models.py
