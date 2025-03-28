#!/bin/bash
#SBATCH --job-name=check_markers_param_notebooks
#SBATCH --output=logs/check_markers_param_notebooks_%a.out
#SBATCH --error=logs/check_markers_param_notebooks_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --account=kubacki.michal
#SBATCH --partition=workq
#SBATCH --array=0-7

# Load necessary modules
source /opt/common/tools/ric.cosr/miniconda3/bin/activate
conda activate snakemake

# Set working directory
cd /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/check_markers

# Make sure logs directory exists
mkdir -p logs
# Create results directory structure
mkdir -p results
mkdir -p notebooks

# Define the arrays of samples and models
SAMPLES=(Emx1_Ctrl Emx1_Mut Nestin_Ctrl Nestin_Mut)
MODELS=(Mouse_Dentate_Gyrus Mouse_Isocortex_Hippocampus)

# Calculate the sample and model indices
SAMPLE_INDEX=$(( SLURM_ARRAY_TASK_ID / ${#MODELS[@]} ))
MODEL_INDEX=$(( SLURM_ARRAY_TASK_ID % ${#MODELS[@]} ))

# Get the current sample and model
CURRENT_SAMPLE=${SAMPLES[$SAMPLE_INDEX]}
CURRENT_MODEL=${MODELS[$MODEL_INDEX]}

echo "Processing array task ID: $SLURM_ARRAY_TASK_ID"
echo "Sample index: $SAMPLE_INDEX, Model index: $MODEL_INDEX"
echo "Processing sample: $CURRENT_SAMPLE"
echo "Processing model: $CURRENT_MODEL"

# Create directory structure for notebooks and results
mkdir -p notebooks/${CURRENT_MODEL}
mkdir -p results/${CURRENT_MODEL}/${CURRENT_SAMPLE}

# Ensure proper permissions for output directory
chmod -R 775 results

# Run the parameterized notebooks script for the current sample-model combination
python run_parameterized_notebooks.py --template manual_annotation.ipynb --samples $CURRENT_SAMPLE --model $CURRENT_MODEL --timeout -1 --force
