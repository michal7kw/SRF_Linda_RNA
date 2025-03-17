# Single-cell RNA-seq Analysis Pipeline for Mouse Brain Nuclear RNA
# This script performs comprehensive analysis of scRNA-seq data from mouse brain
# comparing control vs mutant conditions in Emx1 and Nestin populations

# Load required libraries
library(Seurat)
library(dplyr)
library(ggplot2)
library(patchwork)

# Set random seed for reproducibility
set.seed(42)

# Create output directories
dir.create("results", showWarnings = FALSE)
dir.create("results/qc", showWarnings = FALSE)
dir.create("results/figures", showWarnings = FALSE)

# Function to load and process a single sample
process_sample <- function(sample_path, project_name) {
    # Read data
    data <- Read10X(sample_path)
    
    # Create Seurat object
    seurat_obj <- CreateSeuratObject(
        counts = data,
        project = project_name,
        min.cells = 3,
        min.features = 200
    )
    
    # Calculate mitochondrial percentage
    seurat_obj[['percent.mt']] <- PercentageFeatureSet(seurat_obj, pattern = '^mt-')
    
    return(seurat_obj)
}

# Load all samples
sample_paths <- c(
    'R26_Emx1_Ctrl_adult/outs/filtered_feature_bc_matrix',
    'R26_Emx1_Mut_adult/outs/filtered_feature_bc_matrix',
    'R26_Nestin_Ctrl_adult/outs/filtered_feature_bc_matrix',
    'R26_Nestin_Mut_adult/outs/filtered_feature_bc_matrix'
)

sample_names <- c('Emx1_Ctrl', 'Emx1_Mut', 'Nestin_Ctrl', 'Nestin_Mut')

# Process each sample
seurat_objects <- list()
for (i in seq_along(sample_paths)) {
    if (dir.exists(sample_paths[i])) {
        seurat_objects[[sample_names[i]]] <- process_sample(sample_paths[i], sample_names[i])
    } else {
        warning(paste('Directory not found:', sample_paths[i]))
    }
}

# Add metadata
for (i in seq_along(seurat_objects)) {
    seurat_objects[[i]]$condition <- ifelse(grepl('Ctrl', names(seurat_objects)[i]), 'control', 'mutant')
    seurat_objects[[i]]$celltype <- ifelse(grepl('Emx1', names(seurat_objects)[i]), 'Emx1', 'Nestin')
}

# Merge all samples
seurat_combined <- merge(seurat_objects[[1]],
                        y = seurat_objects[2:length(seurat_objects)],
                        add.cell.ids = sample_names)

# Quality control
VlnPlot(seurat_combined,
        features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
        ncol = 3,
        pt.size = 0.1) +
    ggsave("results/qc/qc_metrics.pdf", width = 12, height = 6)

# Filter cells
seurat_combined <- subset(seurat_combined,
                         subset = nFeature_RNA > 200 &
                                 nFeature_RNA < 6000 &
                                 percent.mt < 10)

# Normalize data
seurat_combined <- NormalizeData(seurat_combined)

# Find variable features
seurat_combined <- FindVariableFeatures(seurat_combined,
                                       selection.method = "vst",
                                       nfeatures = 2000)

# Scale data
all.genes <- rownames(seurat_combined)
seurat_combined <- ScaleData(seurat_combined, features = all.genes)

# Run PCA
seurat_combined <- RunPCA(seurat_combined, features = VariableFeatures(object = seurat_combined))

# Determine dimensionality
ElbowPlot(seurat_combined) +
    ggsave("results/figures/elbow_plot.pdf", width = 8, height = 6)

# Run UMAP and clustering
seurat_combined <- RunUMAP(seurat_combined, dims = 1:30)
seurat_combined <- FindNeighbors(seurat_combined, dims = 1:30)
seurat_combined <- FindClusters(seurat_combined, resolution = 0.5)

# Visualization
DimPlot(seurat_combined, reduction = "umap", group.by = "celltype") +
    ggsave("results/figures/umap_celltype.pdf", width = 10, height = 8)

DimPlot(seurat_combined, reduction = "umap", group.by = "condition") +
    ggsave("results/figures/umap_condition.pdf", width = 10, height = 8)

DimPlot(seurat_combined, reduction = "umap", label = TRUE) +
    ggsave("results/figures/umap_clusters.pdf", width = 10, height = 8)

# Find markers for each cluster
markers <- FindAllMarkers(seurat_combined,
                         only.pos = TRUE,
                         min.pct = 0.25,
                         logfc.threshold = 0.25)

# Save markers to file
write.csv(markers, "results/cluster_markers.csv")

# Top markers heatmap
top10 <- markers %>%
    group_by(cluster) %>%
    top_n(n = 10, wt = avg_log2FC)

DoHeatmap(seurat_combined, features = top10$gene) +
    ggsave("results/figures/top_markers_heatmap.pdf", width = 15, height = 12)

# Differential expression analysis between conditions for each cell type
for (cell_type in c("Emx1", "Nestin")) {
    de_results <- FindMarkers(seurat_combined,
                              ident.1 = "mutant",
                              ident.2 = "control",
                              group.by = "condition",
                              subset.ident = cell_type)
    
    write.csv(de_results,
              file = paste0("results/", cell_type, "_differential_expression.csv"))
}

# Save the Seurat object
saveRDS(seurat_combined, "results/seurat_combined.rds")

# Session info for reproducibility
sink("results/session_info.txt")
print(sessionInfo())
sink()
