#!/bin/bash
#SBATCH --job-name=cell_typist
#SBATCH --output=logs/cell_typist_%a.out
#SBATCH --error=logs/cell_typist_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --account=kubacki.michal
#SBATCH --partition=workq
#SBATCH --array=0-7

# Load necessary modules (adjust as needed for your cluster)
source /opt/common/tools/ric.cosr/miniconda3/bin/activate
conda activate snakemake 

# Set working directory
cd /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/cell_typist

# Define the array of notebook names
notebooks=(
    "annotation_Mouse_Dentate_Gyrus_Emx1_Ctr.ipynb"
    "annotation_Mouse_Dentate_Gyrus_Nestin_Ctr.ipynb"
    "annotation_Mouse_Isocortex_Hippocampus_Emx1_Ctrl.ipynb"
    "annotation_Mouse_Isocortex_Hippocampus_Nestin_Ctrl.ipynb"
    "annotation_Mouse_Dentate_Gyrus_Emx1_Mut.ipynb"
    "annotation_Mouse_Dentate_Gyrus_Nestin_Mut.ipynb"
    "annotation_Mouse_Isocortex_Hippocampus_Emx1_Mut.ipynb"
    "annotation_Mouse_Isocortex_Hippocampus_Nestin_Mut.ipynb"
)

# Get the current notebook based on array task ID
current_notebook=${notebooks[$SLURM_ARRAY_TASK_ID]}

# Check if the file exists
if [ ! -f "$current_notebook" ]; then
    echo "Error: File '$current_notebook' not found"
    exit 1
fi

# Execute the notebook
echo "Executing notebook: $current_notebook"
jupyter nbconvert --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.allow_errors=False \
    --ClearOutputPreprocessor.enabled=False \
    "$current_notebook" 