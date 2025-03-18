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

# Find existing output directories to determine progress
echo "Checking for existing output directories..."
ls -la $OUTPUT_DIR
echo "---------------------------------------"

# Sample names and their processing status
declare -A SAMPLES
SAMPLES["R26_Emx1_Ctrl_adult"]=1
SAMPLES["R26_Emx1_Mut_adult"]=1
SAMPLES["R26_Nestin_Ctrl_adult"]=1
SAMPLES["R26_Nestin_Mut_adult"]=1

# Ask which samples to process
echo ""
echo "Which samples do you want to process? (Enter comma-separated list, or 'all')"
echo "Available samples: R26_Emx1_Ctrl_adult, R26_Emx1_Mut_adult, R26_Nestin_Ctrl_adult, R26_Nestin_Mut_adult"
read -p "Samples to process: " SELECTED_SAMPLES

if [[ "$SELECTED_SAMPLES" != "all" ]]; then
    # Reset all samples to not process
    for SAMPLE in "${!SAMPLES[@]}"; do
        SAMPLES[$SAMPLE]=0
    done
    
    # Set selected samples to process
    IFS=',' read -ra SAMPLE_LIST <<< "$SELECTED_SAMPLES"
    for SAMPLE in "${SAMPLE_LIST[@]}"; do
        SAMPLE=$(echo $SAMPLE | xargs)  # Trim whitespace
        if [[ -n "${SAMPLES[$SAMPLE]}" ]]; then
            SAMPLES[$SAMPLE]=1
        else
            echo "Warning: Unknown sample '$SAMPLE' - ignoring"
        fi
    done
fi

# Run Cell Ranger for selected samples
for SAMPLE in "${!SAMPLES[@]}"; do
    if [[ ${SAMPLES[$SAMPLE]} -eq 1 ]]; then
        echo "====================================="
        echo "Processing sample: $SAMPLE"
        echo "====================================="
        
        # Use a consistent ID for each sample (no timestamp) to allow resuming
        UNIQUE_ID="${SAMPLE}"
        
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
        
        # Create symbolic links for Lane 1
        ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_L001_R1_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_L001_R1_001.fastq.gz"
        ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_L001_R2_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_L001_R2_001.fastq.gz"
        
        # Create symbolic links for Lane 2
        ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_L002_R1_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_L002_R1_001.fastq.gz"
        ln -sf "$DATA_DIR/Sample_${SAMPLE}/${SAMPLE}_${SAMPLE_NUM}_L002_R2_001.fastq.gz" "$FASTQ_DIR/${SAMPLE}_${SAMPLE_NUM}_L002_R2_001.fastq.gz"
        
        # Check if the output directory already exists
        if [[ -d "$UNIQUE_ID" ]]; then
            echo "Found existing output directory: $UNIQUE_ID"
            echo "Will attempt to resume processing..."
            
            # Run Cell Ranger count with --resume flag to continue from previous steps
            $CELLRANGER count \
                --id=${UNIQUE_ID} \
                --transcriptome=$REF \
                --fastqs=$FASTQ_DIR \
                --sample=$SAMPLE \
                --expect-cells=5000 \
                --chemistry=SC3Pv3 \
                --create-bam=true \
                --localcores=14 \
                --localmem=28 \
                --resume
        else
            echo "Starting new processing for sample: $SAMPLE"
            
            # Run Cell Ranger count without resume flag for new processing
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
        fi
        
        # Check if processing was successful
        if [[ $? -eq 0 ]]; then
            echo "Processing completed successfully for sample: $SAMPLE"
        else
            echo "ERROR: Processing failed for sample: $SAMPLE"
            echo "Check logs for details."
        fi
    fi
done

echo "Cell Ranger processing complete!"
echo "Output directories:"
ls -la 