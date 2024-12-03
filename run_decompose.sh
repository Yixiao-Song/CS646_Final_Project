#!/bin/bash -l
#SBATCH -J decomp
#SBATCH --partition=gpu-preempt
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mem=100GB
#SBATCH --constraint=a100-80g
#SBATCH --time=1:00:00
#SBATCH --qos=short
#SBATCH --output=slurm/%x-%j.out
#SBATCH -c 8
#SBATCH -d singleton
#SBATCH --exclude=uri-gpu007
#SBATCH -o /work/pi_miyyer_umass_edu/yapeichang/CS646_Final_Project/logs/%x-%j.out

module load uri/main
module load Java/21.0.2
source /project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/final_proj_env/bin/activate
cd /work/pi_miyyer_umass_edu/yapeichang/CS646_Final_Project

python3 code/decompose.py
