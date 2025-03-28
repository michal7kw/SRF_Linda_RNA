#!/bin/bash
#SBATCH --job-name=cellranger_mut_adult
#SBATCH --output=logs/cellranger_mut_adult_%j.out
#SBATCH --error=logs/cellranger_mut_adult_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=164G
#SBATCH --time=120:00:00
#SBATCH --account=kubacki.michal
#SBATCH --partition=workq

# Set up cleanup trap
cleanup() {
    echo "Cleaning up processes..."
    killall -9 cellranger 2>/dev/null
    echo "Cleanup complete."
    exit
}

# Uncomment these to prevent hard link errors
export CELLRANGER_COPY_MODE=copy
export CELLRANGER_USE_HARDLINKS=false

# Trap signals
trap cleanup SIGINT SIGTERM EXIT

# Create logs directory if it doesn't exist
mkdir -p logs

# Path to Cell Ranger
CELLRANGER="/beegfs/scratch/ric.broccoli/kubacki.michal/tools/cellranger/cellranger-9.0.0/cellranger"

# Path to data directory - USING THE RNA DATA PATH
DATA_DIR="/beegfs/scratch/ric.broccoli/kubacki.michal/Azenta_projects/250307_A00626_0942_BHV7KVDMXY_1/Project_SessaA_2368_Rent_Novaseq6000_w_reagents_scRNA"

# Path to reference genome
REF="/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/refdata-gex-GRCm39-2024-A"

# Output directory
OUTPUT_DIR="/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/cellranger_output_counts_multiome"
mkdir -p $OUTPUT_DIR

# Only processing R26_Emx1_Mut_adult sample
SAMPLE="R26_Emx1_Mut_adult"
echo "Processing sample: $SAMPLE"

# Generate a timestamp for unique IDs
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Create a unique ID for this run
UNIQUE_ID="${SAMPLE}_${TIMESTAMP}"

# Create a directory for fastq files
FASTQ_DIR="$OUTPUT_DIR/${SAMPLE}_fastq"
mkdir -p $FASTQ_DIR

# Sample number for R26_Emx1_Mut_adult is S2
SAMPLE_NUM="S2"

# Create symbolic links for all necessary files
for LANE in "L001" "L002"; do
    ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_R1_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_${LANE}_R1_001.fastq.gz"
    ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_R2_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_${LANE}_R2_001.fastq.gz"
    
    # Link index files (I1 and I2) which are required for multiome data
    if [ -f "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I1_001.fastq.gz" ]; then
        ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I1_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I1_001.fastq.gz"
    fi
    
    if [ -f "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I2_001.fastq.gz" ]; then
        ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I2_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_${LANE}_I2_001.fastq.gz"
    fi
done

# Get number of physical cores on the node for optimal performance
CORES=$SLURM_CPUS_PER_TASK
# Calculate memory in GB, leaving some overhead for the system
MEM_GB=$((${SLURM_MEM_PER_NODE} / 1024 - 16))

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

# Create a local temporary directory
LOCAL_TMP="/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/tmp/cellranger_counts_${SAMPLE}"
mkdir -p $LOCAL_TMP

# Clean up any existing pipestance directory that might conflict
if [ -d "${UNIQUE_ID}" ]; then
    echo "Removing existing pipestance directory: ${UNIQUE_ID}"
    rm -rf "${UNIQUE_ID}"
fi

# Also check in the temp directory
if [ -d "${LOCAL_TMP}/${UNIQUE_ID}" ]; then
    echo "Removing existing pipestance directory in temp location: ${LOCAL_TMP}/${UNIQUE_ID}"
    rm -rf "${LOCAL_TMP}/${UNIQUE_ID}"
fi

# Run Cell Ranger count command with the ARC-v1 chemistry flag
$CELLRANGER count \
    --id=${UNIQUE_ID} \
    --transcriptome=$REF \
    --fastqs=$FASTQ_DIR \
    --sample=$SAMPLE \
    --expect-cells=5000 \
    --chemistry=ARC-v1 \
    --create-bam=true \
    --localcores=$CORES \
    --localmem=$MEM_GB \
    --disable-ui \
    --nosecondary \
    --output-dir=$LOCAL_TMP

# Check if Cell Ranger completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Cell Ranger failed for sample: $SAMPLE"
    echo "Check logs for more details"
    exit 1
fi

# Only proceed with copying if the output directory exists
if [ -d "$LOCAL_TMP/${UNIQUE_ID}/outs/filtered_feature_bc_matrix" ]; then
    echo "Creating a separate directory for count matrices only"
    COUNT_DIR="$OUTPUT_DIR/${SAMPLE}_counts"
    ensure_directory $COUNT_DIR

    # Copy only the essential count matrix files
    rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/filtered_feature_bc_matrix/ $COUNT_DIR/filtered_feature_bc_matrix/
    rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/raw_feature_bc_matrix/ $COUNT_DIR/raw_feature_bc_matrix/
    rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/metrics_summary.csv $COUNT_DIR/
    rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/molecule_info.h5 $COUNT_DIR/
    rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/web_summary.html $COUNT_DIR/

    # Optionally, only copy the full output if needed
    rsync -av $LOCAL_TMP/${UNIQUE_ID}/ $OUTPUT_DIR/${UNIQUE_ID}/
else
    echo "ERROR: Cell Ranger output directory not found at $LOCAL_TMP/${UNIQUE_ID}/outs"
    echo "Cell Ranger likely failed. Check logs for details."
    exit 1
fi

echo "Cell Ranger count processing complete for sample: $SAMPLE"
echo "Count matrices available at: $COUNT_DIR" 