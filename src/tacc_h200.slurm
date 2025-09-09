#!/bin/bash
#SBATCH --partition=gh
#SBATCH --job-name=generate_cot
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH -t 48:00:00
#SBATCH --output=tacc/out/%x_%j.out
#SBATCH --error=tacc/err/%x_%j.err
#SBATCH --account=ASC24077
#SBATCH --mail-type=all
#SBATCH --mail-user=acnagle@utexas.edu

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export NCCL_NET_GDR_LEVEL="SYS"
export NCCL_NET_GDR_READ=1

export HF_HOME=$WORK/hf_cache

export CONDA_BASE=$HOME/micromamba

export PATH="$CONDA_BASE/condabin:$PATH"
source $CONDA_BASE/etc/profile.d/micromamba.sh

module load gcc
module load cuda/12.6

micromamba activate reason

srun bash run.sh
