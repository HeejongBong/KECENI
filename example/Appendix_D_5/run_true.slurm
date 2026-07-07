#!/bin/bash
#SBATCH --job-name=run_true
#SBATCH --chdir=.
#SBATCH --output=./logs/%x-%A_%a.log
#SBATCH --error=./errs/%x-%A_%a.err
#SBATCH --mail-user=bong0@purdue.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-79
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --partition=cpu
#SBATCH --account=bong0

# -----------------------------
# Load environment and modules
# -----------------------------
module load conda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate causalnet-negishi

# -----------------------------
# Working directory
# -----------------------------
mkdir -p logs errs

# -----------------------------
# Run the R script
# -----------------------------
python run_true.py $SLURM_ARRAY_TASK_ID