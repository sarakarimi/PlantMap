#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32G
#SBATCH --account=Berzelius-2024-465

export NUM_AGENTS=6
export AGENT_ID="$1"

uv run fedn client start -in client_files/client"$1".yaml --secure=True --force-ssl 