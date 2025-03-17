# scRNA-seq Analysis Pipeline
# Mouse Brain Nuclear RNA Analysis
# Date: 2024-03-14

# Load required libraries
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

required_packages <- c(
    "Seurat",
    "tidyverse",
    "Matrix",
    "ggplot2",
    "cowplot",
    "patchwork",
    "DoubletFinder"
)

# Install missing packages
for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE)) {
        if (pkg %in% c("Seurat", "tidyverse", "Matrix", "ggplot2", "cowplot", "patchwork")) {
            install.packages(pkg)
        } else if (pkg == "DoubletFinder") {
            remotes::install_github('chris-mcginnis-ucsf/DoubletFinder')
        }
    }
}

# Load libraries
library(Seurat)
library(tidyverse)
library(Matrix)
library(ggplot2)
library(cowplot)
library(patchwork)
library(DoubletFinder)

# Set working directory and paths
# This is the path to the Cell Ranger output
data_dir <- "/home/michal/SRF/Linda/RNA/cellranger_output"
output_dir <- "analysis/results"
figure_dir <- "analysis/figures"

# Create directories if they don't exist
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

# Function to process each sample
process_sample <- function(sample_path, sample_name) {
    # Read the Cell Ranger output
    # Cell Ranger output directory structure: sample_id/outs/filtered_feature_bc_matrix
    data <- Read10X(data.dir = file.path(sample_path, "outs", "filtered_feature_bc_matrix"))
    
    # Create Seurat object
    seurat_obj <- CreateSeuratObject(
        counts = data,
        project = sample_name,
        min.cells = 3,
        min.features = 200
    )
    
    # Add metadata
    seurat_obj$sample <- sample_name
    seurat_obj$condition <- ifelse(grepl("Ctrl", sample_name), "Control", "Mutant")
    seurat_obj$gene_group <- ifelse(grepl("Emx1", sample_name), "Emx1", "Nestin")
    
    # Calculate mitochondrial percentage
    seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^mt-")
    
    return(seurat_obj)
}

# List of samples
samples <- list(
    "R26_Emx1_Ctrl" = "R26_Emx1_Ctrl_adult",
    "R26_Emx1_Mut" = "R26_Emx1_Mut_adult",
    "R26_Nestin_Ctrl" = "R26_Nestin_Ctrl_adult",
    "R26_Nestin_Mut" = "R26_Nestin_Mut_adult"
)

# Process all samples
seurat_objects <- list()
for (sample_name in names(samples)) {
    sample_path <- file.path(data_dir, samples[[sample_name]])
    print(paste("Processing", sample_name))
    seurat_objects[[sample_name]] <- process_sample(sample_path, sample_name)
}

# Function for QC plots
generate_qc_plots <- function(seurat_obj, sample_name) {
    # Violin plots
    vln_plot <- VlnPlot(seurat_obj, 
                        features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
                        ncol = 3)
    
    # Feature scatter plots
    scatter1 <- FeatureScatter(seurat_obj, 
                              feature1 = "nCount_RNA", 
                              feature2 = "nFeature_RNA")
    scatter2 <- FeatureScatter(seurat_obj, 
                              feature1 = "nCount_RNA", 
                              feature2 = "percent.mt")
    
    # Combine plots
    combined_plot <- vln_plot / (scatter1 | scatter2)
    
    # Save plot
    ggsave(
        filename = file.path(figure_dir, paste0(sample_name, "_qc_plots.pdf")),
        plot = combined_plot,
        width = 12,
        height = 10
    )
    
    return(combined_plot)
}

# Generate QC plots for each sample
qc_plots <- list()
for (sample_name in names(seurat_objects)) {
    print(paste("Generating QC plots for", sample_name))
    qc_plots[[sample_name]] <- generate_qc_plots(seurat_objects[[sample_name]], sample_name)
}

# Function to filter cells
filter_cells <- function(seurat_obj) {
    seurat_obj <- subset(seurat_obj, 
                        subset = nFeature_RNA > 200 &
                                nFeature_RNA < 6000 &
                                percent.mt < 20)
    return(seurat_obj)
}

# Filter all samples
filtered_objects <- lapply(seurat_objects, filter_cells)

# Normalize and find variable features for each dataset
for (i in seq_along(filtered_objects)) {
    filtered_objects[[i]] <- NormalizeData(filtered_objects[[i]])
    filtered_objects[[i]] <- FindVariableFeatures(filtered_objects[[i]], 
                                                selection.method = "vst",
                                                nfeatures = 2000)
}

# Save the processed objects
saveRDS(filtered_objects, file = file.path(output_dir, "processed_seurat_objects.rds"))

# Print processing summary
print("Processing complete. Filtered Seurat objects have been saved.")
print("Next steps will include:")
print("1. Integration of samples")
print("2. Dimensionality reduction")
print("3. Clustering")
print("4. Differential expression analysis")
print("5. Cell type annotation")

# The next part of the analysis will be implemented after reviewing the QC results 