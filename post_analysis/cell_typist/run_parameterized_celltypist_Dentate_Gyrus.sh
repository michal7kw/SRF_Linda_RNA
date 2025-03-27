#!/bin/bash
#SBATCH --job-name=celltypist_Dentate_Gyrus
#SBATCH --output=logs/celltypist_Dentate_Gyrus_%a.out
#SBATCH --error=logs/celltypist_Dentate_Gyrus_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=kubacki.michal
#SBATCH --partition=workq
#SBATCH --array=0-7

# Load necessary modules
source /opt/common/tools/ric.cosr/miniconda3/bin/activate
conda activate snakemake

# Set working directory
cd /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/cell_typist

# Make sure logs directory exists
mkdir -p logs
mkdir -p notebooks

# Define the array of models and samples
MODELS=("Mouse_Dentate_Gyrus") # "Mouse_Isocortex_Hippocampus"
SAMPLES=("Emx1_Ctrl" "Emx1_Mut" "Nestin_Ctrl" "Nestin_Mut")

# Calculate the model and sample indices based on the task ID
# Each sample will be run for each model, so we have MODEL_COUNT * SAMPLE_COUNT combinations
MODEL_COUNT=${#MODELS[@]}
SAMPLE_COUNT=${#SAMPLES[@]}
TOTAL_COMBINATIONS=$((MODEL_COUNT * SAMPLE_COUNT))

# Check if SLURM_ARRAY_TASK_ID is within bounds
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_COMBINATIONS" ]; then
    echo "Error: Task ID $SLURM_ARRAY_TASK_ID is out of bounds (0-$((TOTAL_COMBINATIONS-1)))"
    exit 1
fi

# Calculate which model and sample to use for this task
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / SAMPLE_COUNT))
SAMPLE_INDEX=$((SLURM_ARRAY_TASK_ID % SAMPLE_COUNT))

CURRENT_MODEL=${MODELS[$MODEL_INDEX]}
CURRENT_SAMPLE=${SAMPLES[$SAMPLE_INDEX]}

echo "Processing model: $CURRENT_MODEL, sample: $CURRENT_SAMPLE (Task ID: $SLURM_ARRAY_TASK_ID)"

# Run the parameterized notebook generator for the current model-sample combination
python run_parameterized_celltypist.py --template template_annotation_Mouse_Dentate_Gyrus.ipynb --model $CURRENT_MODEL --sample $CURRENT_SAMPLE --timeout 86400 --force --output-dir notebooks 