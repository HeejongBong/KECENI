#!/bin/bash
#SBATCH --job-name=run_misspi
#SBATCH --output=/home/%u/KECENI_anon/example/Appendix_D_5/logs/%x-%A_%a.log
#SBATCH --error=/home/%u/KECENI_anon/example/Appendix_D_5/errs/%x-%A_%a.err
#SBATCH --array=0-79
#SBATCH --time=01:00:00
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
cd ~/KECENI_anon/example/Appendix_D_5

python run_misspi.py $SLURM_ARRAY_TASK_ID
