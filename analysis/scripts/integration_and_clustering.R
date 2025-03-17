# Integration and Downstream Analysis
# Mouse Brain Nuclear RNA Analysis
# Date: 2024-03-14

# Load required libraries
library(Seurat)
library(tidyverse)
library(ggplot2)
library(patchwork)

# Load the processed objects
output_dir <- "analysis/results"
figure_dir <- "analysis/figures"
filtered_objects <- readRDS(file.path(output_dir, "processed_seurat_objects.rds"))

# Select features for integration
features <- SelectIntegrationFeatures(object.list = filtered_objects)

# Find integration anchors
anchors <- FindIntegrationAnchors(
    object.list = filtered_objects,
    anchor.features = features
)

# Create integrated dataset
seurat_integrated <- IntegrateData(
    anchorset = anchors,
    dims = 1:30
)

# Switch to integrated assay for downstream analysis
DefaultAssay(seurat_integrated) <- "integrated"

# Scale data and run PCA
seurat_integrated <- ScaleData(seurat_integrated, verbose = FALSE)
seurat_integrated <- RunPCA(seurat_integrated, verbose = FALSE)

# Generate elbow plot
elbow_plot <- ElbowPlot(seurat_integrated, ndims = 50)
ggsave(file.path(figure_dir, "elbow_plot.pdf"), elbow_plot)

# Run UMAP and clustering
seurat_integrated <- RunUMAP(seurat_integrated, dims = 1:30)
seurat_integrated <- FindNeighbors(seurat_integrated, dims = 1:30)
seurat_integrated <- FindClusters(seurat_integrated, resolution = 0.8)

# Generate UMAP visualization plots
umap_condition <- DimPlot(seurat_integrated, 
                         reduction = "umap", 
                         group.by = "condition",
                         cols = c("Control" = "#F8766D", "Mutant" = "#00BFC4"))

umap_clusters <- DimPlot(seurat_integrated, 
                        reduction = "umap", 
                        label = TRUE)

umap_gene_group <- DimPlot(seurat_integrated, 
                          reduction = "umap", 
                          group.by = "gene_group",
                          cols = c("Emx1" = "#7CAE00", "Nestin" = "#C77CFF"))

# Combine and save UMAP plots
combined_umaps <- umap_condition + umap_clusters + umap_gene_group
ggsave(file.path(figure_dir, "combined_umaps.pdf"), combined_umaps, width = 15, height = 5)

# Find markers for each cluster
DefaultAssay(seurat_integrated) <- "RNA"
markers <- FindAllMarkers(
    seurat_integrated,
    only.pos = TRUE,
    min.pct = 0.25,
    logfc.threshold = 0.25
)

# Save markers
write.csv(markers, file.path(output_dir, "cluster_markers.csv"))

# Generate top markers heatmap
top10 <- markers %>%
    group_by(cluster) %>%
    top_n(n = 10, wt = avg_log2FC)

heatmap_plot <- DoHeatmap(
    seurat_integrated,
    features = unique(top10$gene),
    group.by = "seurat_clusters"
) + NoLegend()

ggsave(file.path(figure_dir, "markers_heatmap.pdf"), heatmap_plot, width = 12, height = 15)

# Save the integrated object
saveRDS(seurat_integrated, file.path(output_dir, "integrated_seurat_object.rds"))

# Differential expression analysis between conditions
# For each gene group (Emx1 and Nestin)
gene_groups <- unique(seurat_integrated$gene_group)

for (group in gene_groups) {
    # Subset the data for the current gene group
    subset_data <- subset(seurat_integrated, gene_group == group)
    
    # Find DEGs between conditions
    Idents(subset_data) <- "condition"
    degs <- FindMarkers(
        subset_data,
        ident.1 = "Mutant",
        ident.2 = "Control",
        min.pct = 0.25,
        logfc.threshold = 0.25
    )
    
    # Save DEGs
    write.csv(degs, file.path(output_dir, paste0("DEGs_", group, ".csv")))
    
    # Generate volcano plot
    volcano_data <- degs %>%
        rownames_to_column("gene") %>%
        mutate(
            significant = p_val_adj < 0.05 & abs(avg_log2FC) > 0.5,
            label = ifelse(significant & abs(avg_log2FC) > 1, gene, "")
        )
    
    volcano_plot <- ggplot(volcano_data, aes(x = avg_log2FC, y = -log10(p_val_adj))) +
        geom_point(aes(color = significant)) +
        geom_text_repel(aes(label = label), max.overlaps = 20) +
        theme_minimal() +
        labs(title = paste("Volcano Plot -", group),
             x = "log2 Fold Change",
             y = "-log10 Adjusted p-value")
    
    ggsave(file.path(figure_dir, paste0("volcano_plot_", group, ".pdf")), 
           volcano_plot, 
           width = 10, 
           height = 8)
}

print("Integration and downstream analysis complete. Results have been saved.")
print("Generated files:")
print("1. Integrated Seurat object")
print("2. Cluster markers")
print("3. Differential expression results")
print("4. Various visualization plots") 