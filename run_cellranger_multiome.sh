# Mystery Solved: You Have Multiome Data (ATAC+RNA)

This log message reveals the exact problem:

> "Cell Ranger detected the chemistry **ARC-v1**, which may indicate a workflow error during sample preparation."

## What's Happening

1. **You have 10x Multiome data** (combined ATAC+RNA from the same cells)
   - ARC-v1 is the chemistry for 10x Genomics Chromium Single Cell Multiome ATAC + Gene Expression
   - This requires a completely different analysis pipeline than standard scRNA-seq

2. **Your current script is using the standard RNA-seq pipeline**
   - The standard `cellranger count` command doesn't understand multiome barcode structures
   - This explains the 3.6% valid barcodes - the barcode structures are different

## How to Fix This

You need to use Cell Ranger ARC, which is specifically designed for multiome data:

```bash
#!/bin/bash
#SBATCH --job-name=cellranger_arc
#SBATCH --output=logs/cellranger_arc_%A_%a.out
#SBATCH --error=logs/cellranger_arc_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=164G
#SBATCH --time=120:00:00
#SBATCH --account=kubacki.michal
#SBATCH --partition=workq

# Set up cleanup trap
cleanup() {
    echo "Cleaning up processes..."
    killall -9 cellranger-arc 2>/dev/null
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

# Path to Cell Ranger ARC
CELLRANGER_ARC="/beegfs/scratch/ric.broccoli/kubacki.michal/tools/cellranger/cellranger-arc-2.0.0/cellranger-arc"

# Path to data directory
DATA_DIR="/beegfs/scratch/ric.broccoli/kubacki.michal/Azenta_projects/250307_A00626_0942_BHV7KVDMXY_1/Project_SessaA_2367_Rent_Novaseq6000_w_reagents_scATAC"

# Path to reference genome
REF="/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/refdata-cellranger-arc-mm10-2020-A"

# Output directory
OUTPUT_DIR="/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/cellranger_arc_output"
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
    
    # Also link index files (I1 and I2) which are required for multiome data
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

echo "Using $CORES cores and ${MEM_GB}GB memory for Cell Ranger ARC"

# Add this function before running cellranger-arc
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
LOCAL_TMP="/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/tmp/cellranger_arc_${SAMPLE}_${SLURM_ARRAY_TASK_ID}"
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

# Run Cell Ranger ARC count command - explicitly specify ARC-v1 chemistry
$CELLRANGER_ARC count \
    --id=${UNIQUE_ID} \
    --reference=$REF \
    --libraries=/path/to/libraries.csv \
    --chemistry=ARC-v1 \
    --localcores=$CORES \
    --localmem=$MEM_GB \
    --disable-ui \
    --output-dir=$LOCAL_TMP

# Check if Cell Ranger ARC completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Cell Ranger ARC failed for sample: $SAMPLE"
    echo "Check logs for more details"
    exit 1
fi

# Only proceed with copying if the output directory exists
if [ -d "$LOCAL_TMP/${UNIQUE_ID}/outs" ]; then
    echo "Creating a separate directory for count matrices only"
    COUNT_DIR="$OUTPUT_DIR/${SAMPLE}_counts"
    ensure_directory $COUNT_DIR

    # Copy relevant output files for both modalities
    # RNA output
    if [ -d "$LOCAL_TMP/${UNIQUE_ID}/outs/filtered_feature_bc_matrix" ]; then
        rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/filtered_feature_bc_matrix/ $COUNT_DIR/filtered_feature_bc_matrix/
        rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/raw_feature_bc_matrix/ $COUNT_DIR/raw_feature_bc_matrix/
    fi
    
    # ATAC output
    if [ -d "$LOCAL_TMP/${UNIQUE_ID}/outs/filtered_peak_bc_matrix" ]; then
        rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/filtered_peak_bc_matrix/ $COUNT_DIR/filtered_peak_bc_matrix/
        rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/raw_peak_bc_matrix/ $COUNT_DIR/raw_peak_bc_matrix/
    fi
    
    # Summary files
    rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/metrics_summary.csv $COUNT_DIR/
    rsync -av $LOCAL_TMP/${UNIQUE_ID}/outs/web_summary.html $COUNT_DIR/
    
    # Optionally, copy the full output if needed
    rsync -av $LOCAL_TMP/${UNIQUE_ID}/ $OUTPUT_DIR/${UNIQUE_ID}/
else
    echo "ERROR: Cell Ranger ARC output directory not found at $LOCAL_TMP/${UNIQUE_ID}/outs"
    echo "Cell Ranger ARC likely failed. Check logs for details."
    exit 1
fi

echo "Cell Ranger ARC processing complete for sample: $SAMPLE"
echo "Count matrices available at: $COUNT_DIR"

```

## Important: Create a Libraries CSV File

For multiome data, you need to create a libraries.csv file with the following structure:

```csv
fastqs,sample,library_type
/path/to/fastqs/gex,R26_Emx1_Ctrl_adult,Gene Expression
/path/to/fastqs/atac,R26_Emx1_Ctrl_adult,Chromatin Accessibility
```

This tells Cell Ranger ARC which FASTQ files are for gene expression and which are for chromatin accessibility.

## Additional Considerations

1. **Reference Genome**: You need a special multiome reference like `refdata-cellranger-arc-mm10-2020-A`

2. **Software Installation**: Make sure you have Cell Ranger ARC installed, not just standard Cell Ranger

3. **Further Information**: Check the 10x Genomics documentation for Multiome data: 
   https://www.10xgenomics.com/support/software/cell-ranger-arc/latest

This changes the entire approach to your analysis, but now you know exactly why your standard RNA-seq pipeline was failing - you're dealing with multiome data that requires a specialized pipeline!