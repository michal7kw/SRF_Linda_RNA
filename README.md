# scRNA-seq Analysis Pipeline for Mouse Brain Nuclear RNA

This repository contains scripts for analyzing single-cell RNA-seq data from mouse brain nuclear RNA samples.

## Data Description

The dataset consists of 4 samples:
- R26_Emx1_Ctrl_adult (Control)
- R26_Emx1_Mut_adult (Mutant)
- R26_Nestin_Ctrl_adult (Control)
- R26_Nestin_Mut_adult (Mutant)

Each sample has been sequenced across 2 lanes (L001 and L002) with paired-end reads (R1 and R2) and index reads (I1 and I2).

## Analysis Workflow

### 1. Cell Ranger Processing

The raw FASTQ files need to be processed using Cell Ranger before analysis with Seurat.

1. Edit the `run_cellranger.sh` script to specify the correct path to your reference genome.
2. Make the script executable:
   ```
   chmod +x run_cellranger.sh
   ```
3. Run the script:
   ```
   ./run_cellranger.sh
   ```

This will process each sample through Cell Ranger and generate count matrices in the `cellranger_output` directory.

### 2. Seurat Analysis

After Cell Ranger processing, the data can be analyzed using Seurat.

1. Initial Processing and QC:
   ```
   Rscript analysis/scripts/scRNA_analysis.R
   ```

2. Integration and Clustering:
   ```
   Rscript analysis/scripts/integration_and_clustering.R
   ```

3. Cell Type Annotation:
   ```
   Rscript analysis/scripts/cell_type_annotation.R
   ```

## Output Files

The analysis will generate several output files:

- `analysis/results/`: Contains processed Seurat objects and analysis results
- `analysis/figures/`: Contains visualization plots

## Requirements

- Cell Ranger (v9.0.1 or later)
- R (v4.0.0 or later)
- Required R packages:
  - Seurat
  - tidyverse
  - Matrix
  - ggplot2
  - cowplot
  - patchwork
  - DoubletFinder

## Notes

- The filtering parameters (nFeature_RNA > 200 & nFeature_RNA < 6000 & percent.mt < 20) may need to be adjusted based on your specific data.
- Cell type markers in the annotation script are based on common markers for mouse brain cell types and may need to be adjusted.
- The expected number of cells per sample is set to 5000 in the Cell Ranger command, which may need to be adjusted based on your experiment. 