#!/bin/bash
#SBATCH --job-name=run_alpha
#SBATCH --output=/home/%u/KECENI_anon/example/Section_5_2/logs/%x-%A_%a.log
#SBATCH --error=/home/%u/KECENI_anon/example/Section_5_2/errs/%x-%A_%a.err
#SBATCH --array=0-79
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=elevina0

# Load necessary modules
module load python3.10-anaconda/2023.03

source $(conda info --base)/etc/profile.d/conda.sh

# Activate virtual environment if needed
conda activate causalnet

# Run the simulation script
cd ~/KECENI_anon/example/Section_5_2

python run_alpha.py $SLURM_ARRAY_TASK_ID
