#!/bin/bash
#SBATCH --job-name=iNET
#SBATCH --output=slurm/debug.out
#SBATCH --error=slurm/debug.out
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
##SBATCH --constraint="geforce_rtx_3090" #"a6000" #
##SBATCH -w n-303 #rack-omerl-g01

source ~/.bashrc
export JUPYTER_RUNTIME_DIR=/home/bnet/kupershmidt/anaconda3/envs/spider/share/jupyter
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export NETWORKX_AUTOMATIC_BACKENDS=cugraph
conda activate spider

jupyter-notebook --no-browser --ip=0.0.0.0 --port=8888
