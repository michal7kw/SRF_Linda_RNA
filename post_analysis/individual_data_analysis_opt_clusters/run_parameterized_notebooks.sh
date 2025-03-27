#!/bin/bash
#SBATCH --job-name=param_notebooks
#SBATCH --output=logs/param_notebooks_%a.out
#SBATCH --error=logs/param_notebooks_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --account=kubacki.michal
#SBATCH --partition=workq
#SBATCH --array=0-3

# Load necessary modules
source /opt/common/tools/ric.cosr/miniconda3/bin/activate
conda activate snakemake

# Set working directory
cd /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters

# Make sure logs directory exists
mkdir -p logs

# Define the array of samples
SAMPLES=(Emx1_Ctrl Emx1_Mut Nestin_Ctrl Nestin_Mut)

# Get the current sample based on the array task ID
CURRENT_SAMPLE=${SAMPLES[$SLURM_ARRAY_TASK_ID]}

echo "Processing sample: $CURRENT_SAMPLE"

# Run the parameterized notebooks script for the current sample only
python run_parameterized_notebooks.py --template template_notebook.ipynb --samples $CURRENT_SAMPLE --timeout -1 --force
