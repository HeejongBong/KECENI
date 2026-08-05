#!/bin/bash
#SBATCH --job-name=run_keceni_kernel
#SBATCH --chdir=.
#SBATCH --output=./logs/%x-%A_%a.log
#SBATCH --error=./errs/%x-%A_%a.err
#SBATCH --mail-user=bong0@purdue.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-79
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --account=statdept

# -----------------------------
# Load environment and modules
# -----------------------------
module load conda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate causalnet

# -----------------------------
# Working directory
# -----------------------------

mkdir -p logs errs

# -----------------------------
# Run
# -----------------------------
python ${SLURM_JOB_NAME}.py ${SLURM_ARRAY_TASK_ID}
