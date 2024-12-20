#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=0-05:00:00
#SBATCH --mem=32G
#SBATCH --partition=rleap_gpu_24gb

export NUM_AGENTS=6
export AGENT_ID="$1"

uv run --cache-dir uv_cache/ fedn client start -in client_files/client"$1".yaml --secure=True --force-ssl --local-package