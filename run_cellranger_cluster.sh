#!/bin/bash
#SBATCH --job-name=cellranger_cluster
#SBATCH --output=logs/cellranger_cluster_%A_%a.out
#SBATCH --error=logs/cellranger_cluster_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --account=kubacki.michal
#SBATCH --partition=workq

export CELLRANGER_COPY_MODE=copy
export CELLRANGER_USE_HARDLINKS=false

# Set up cleanup trap
cleanup() {
    echo "Cleaning up processes..."
    killall -9 cellranger 2>/dev/null
    echo "Cleanup complete."
    exit
}

# Trap signals
trap cleanup SIGINT SIGTERM EXIT

# Create logs directory if it doesn't exist
mkdir -p logs

# Path to Cell Ranger
CELLRANGER="/beegfs/scratch/ric.broccoli/kubacki.michal/tools/cellranger/cellranger-9.0.0/cellranger"

# Path to data directory
DATA_DIR="/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/250307_A00626_0942_BHV7KVDMXY_1/Project_SessaA_2368_Rent_Novaseq6000_w_reagents_scRNA"

# Path to reference genome
REF="/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/refdata-gex-GRCm39-2024-A"

# Output directory
OUTPUT_DIR="/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/cellranger_output_retry"
mkdir -p $OUTPUT_DIR

# Sample names
SAMPLES=("R26_Emx1_Ctrl_adult" "R26_Emx1_Mut_adult" "R26_Nestin_Ctrl_adult" "R26_Nestin_Mut_adult")

# Get the current sample based on array task ID
SAMPLE="${SAMPLES[$SLURM_ARRAY_TASK_ID]}"
echo "Processing sample: $SAMPLE"

# Generate a timestamp for unique IDs
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Create a unique ID for this run
UNIQUE_ID="${SAMPLE}_${TIMESTAMP}"

# Create a directory for fastq files
FASTQ_DIR="$OUTPUT_DIR/${SAMPLE}_fastq"
mkdir -p $FASTQ_DIR

# Get the sample number (S1, S2, etc.)
if [[ $SAMPLE == "R26_Emx1_Ctrl_adult" ]]; then
    SAMPLE_NUM="S1"
elif [[ $SAMPLE == "R26_Emx1_Mut_adult" ]]; then
    SAMPLE_NUM="S2"
elif [[ $SAMPLE == "R26_Nestin_Ctrl_adult" ]]; then
    SAMPLE_NUM="S3"
elif [[ $SAMPLE == "R26_Nestin_Mut_adult" ]]; then
    SAMPLE_NUM="S4"
fi

# Create symbolic links for all necessary files
for LANE in "L001" "L002"; do
    ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_R1_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_${LANE}_R1_001.fastq.gz"
    ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_R2_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_${LANE}_R2_001.fastq.gz"
    
    # Also link index files (I1 and I2) if needed by cellranger
    if [ -f "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I1_001.fastq.gz" ]; then
        ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I1_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I1_001.fastq.gz"
    fi
    
    if [ -f "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I2_001.fastq.gz" ]; then
        ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I2_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I2_001.fastq.gz"
    fi
done

# Get number of physical cores on the node for optimal performance
CORES=$(nproc)
# Calculate memory in GB, leaving some overhead for the system
MEM_GB=$(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 - 8))

echo "Using $CORES cores and ${MEM_GB}GB memory for Cell Ranger"

# Add this function before running cellranger
ensure_directory() {
    local max_retries=5
    local count=0
    while [ $count -lt $max_retries ]; do
        mkdir -p "$1" && break
        echo "Failed to create directory $1, retrying in 10s..."
        sleep 10
        count=$((count+1))
    done
    if [ $count -eq $max_retries ]; then
        echo "Failed to create directory after $max_retries attempts"
        return 1
    fi
    return 0
}

# Then use it before each mkdir
ensure_directory $OUTPUT_DIR
ensure_directory $FASTQ_DIR

# Run Cell Ranger count with optimized resource allocation
$CELLRANGER count \
    --id=${UNIQUE_ID} \
    --transcriptome=$REF \
    --fastqs=$FASTQ_DIR \
    --sample=$SAMPLE \
    --expect-cells=5000 \
    --chemistry=SC3Pv3 \
    --create-bam=true \
    --localcores=$CORES \
    --localmem=$MEM_GB \
    --disable-ui

echo "Cell Ranger processing complete for sample: $SAMPLE" 