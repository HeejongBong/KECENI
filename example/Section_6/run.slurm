#!/bin/bash
#SBATCH --output=/home/%u/KECENI_anon/example/Section_6/logs/%x-%A_%a.log
#SBATCH --error=/home/%u/KECENI_anon/example/Section_6/logs/%x-%A_%a.err
#SBATCH --array=0-104
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=stats_dept1

# Load necessary modules
module load python3.10-anaconda/2023.03

source $(conda info --base)/etc/profile.d/conda.sh

# Activate virtual environment if needed
conda activate causalnet

# Run the simulation script
cd ~/KECENI_anon/example/Section_6

python $SLURM_JOB_NAME.py $SLURM_ARRAY_TASK_ID
