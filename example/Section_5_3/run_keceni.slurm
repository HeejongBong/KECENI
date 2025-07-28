#!/bin/bash
#SBATCH --job-name=run_keceni
#SBATCH --output=/home/%u/KECENI_anon/example/Section_5_3/logs/%x-%A_%a.log
#SBATCH --error=/home/%u/KECENI_anon/example/Section_5_3/errs/%x-%A_%a.err
#SBATCH --array=0-39
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --account=elevina0

# Load necessary modules
module load python3.10-anaconda/2023.03

source $(conda info --base)/etc/profile.d/conda.sh

# Activate virtual environment if needed
conda activate causalnet

# Run the simulation script
cd ~/KECENI_anon/example/Section_5_3

python run_keceni.py $SLURM_ARRAY_TASK_ID
