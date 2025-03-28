#!/bin/bash
#SBATCH --job-name=unassigned_Isocortex_Hippocampus
#SBATCH --output=logs/unassigned_Isocortex_Hippocampus_%a.out
#SBATCH --error=logs/unassigned_Isocortex_Hippocampus_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=kubacki.michal
#SBATCH --partition=workq
#SBATCH --array=0-3

# Load necessary modules
source /opt/common/tools/ric.cosr/miniconda3/bin/activate
conda activate snakemake

# Set working directory
cd /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/cell_typist/check_unannotated

# Make sure logs directory exists
mkdir -p logs
mkdir -p unassigned_notebooks

# Define the model and samples
MODEL="Mouse_Isocortex_Hippocampus"
SAMPLES=("Emx1_Ctrl" "Emx1_Mut" "Nestin_Ctrl" "Nestin_Mut")

# Get the current sample based on the array task ID
CURRENT_SAMPLE=${SAMPLES[$SLURM_ARRAY_TASK_ID]}

echo "Processing unassigned cells for model: $MODEL, sample: $CURRENT_SAMPLE (Task ID: $SLURM_ARRAY_TASK_ID)"

# Run the parameterized notebook generator for the current sample
python run_parameterized_unassigned_analysis.py --template template_analyze_unassigned_cells_Isocortex_Hippocampus.ipynb --model $MODEL --sample $CURRENT_SAMPLE --timeout 86400 --force --output-dir unassigned_notebooks --execute 