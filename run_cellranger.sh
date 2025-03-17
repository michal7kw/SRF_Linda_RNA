#!/bin/bash

# Set up cleanup trap
cleanup() {
    echo "Cleaning up processes..."
    killall -9 cellranger 2>/dev/null
    echo "Cleanup complete."
    exit
}

# Trap signals
trap cleanup SIGINT SIGTERM EXIT

# Path to Cell Ranger
CELLRANGER="/home/michal/Download/cellranger-9.0.1/cellranger"

# Path to data directory
DATA_DIR="/home/michal/SRF/Linda/RNA/250307_A00626_0942_BHV7KVDMXY_1/Project_SessaA_2368_Rent_Novaseq6000_w_reagents_scRNA"

# Path to reference genome - UPDATED to use the proper Cell Ranger reference
REF="/home/michal/SRF/Linda/RNA/cellranger_reference/refdata-gex-mm10-2020-A"

# Output directory
OUTPUT_DIR="/home/michal/SRF/Linda/RNA/cellranger_output"
mkdir -p $OUTPUT_DIR

# Sample names
SAMPLES=("R26_Emx1_Ctrl_adult" "R26_Emx1_Mut_adult" "R26_Nestin_Ctrl_adult" "R26_Nestin_Mut_adult")

# Generate a timestamp for unique IDs
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Run Cell Ranger for each sample
for SAMPLE in "${SAMPLES[@]}"; do
    echo "Processing sample: $SAMPLE"
    
    # Create a unique ID for this run
    UNIQUE_ID="${SAMPLE}_${TIMESTAMP}"
    
    # Create a directory for fastq files
    FASTQ_DIR="$OUTPUT_DIR/${SAMPLE}_fastq"
    mkdir -p $FASTQ_DIR
    
    # Create symbolic links to the fastq files with the correct naming convention
    # Cell Ranger expects files named: SAMPLE_S1_L00X_R1_001.fastq.gz, SAMPLE_S1_L00X_R2_001.fastq.gz
    
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
    
    # Create symbolic links for Lane 1
    ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_L001_R1_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_L001_R1_001.fastq.gz"
    ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_L001_R2_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_L001_R2_001.fastq.gz"
    
    # Create symbolic links for Lane 2
    ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_L002_R1_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_L002_R1_001.fastq.gz"
    ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_L002_R2_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_L002_R2_001.fastq.gz"
    
    # Run Cell Ranger count with unique ID and explicitly specified chemistry
    $CELLRANGER count \
        --id=${UNIQUE_ID} \
        --transcriptome=$REF \
        --fastqs=$FASTQ_DIR \
        --sample=$SAMPLE \
        --expect-cells=5000 \
        --chemistry=SC3Pv3 \
        --create-bam=true \
        --localcores=14 \
        --localmem=28
done

echo "Cell Ranger processing complete!" 