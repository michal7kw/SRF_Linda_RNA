#!/bin/bash
#SBATCH --job-name=marker_heatmaps
#SBATCH --output=logs/marker_heatmaps_%a.out
#SBATCH --error=logs/marker_heatmaps_%a.err
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

# Define the processed data file path
PROCESSED_FILE="cellranger_counts_R26_${CURRENT_SAMPLE}_adult_${SLURM_ARRAY_TASK_ID}/${CURRENT_SAMPLE}_processed.h5ad"

# Define the output directory for heatmaps
OUTPUT_DIR="cellranger_counts_R26_${CURRENT_SAMPLE}_adult_${SLURM_ARRAY_TASK_ID}/heatmaps"

# Check if the processed file exists
if [ ! -f "$PROCESSED_FILE" ]; then
    echo "Error: Processed file $PROCESSED_FILE not found"
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the heatmap generation script
python generate_marker_heatmaps.py \
    --adata "$PROCESSED_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --n_genes 10
    # --cluster_specific

echo "Heatmap generation completed for $CURRENT_SAMPLE"