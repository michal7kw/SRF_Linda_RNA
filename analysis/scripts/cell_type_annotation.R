# Cell Type Annotation
# Mouse Brain Nuclear RNA Analysis
# Date: 2024-03-14

# Load required libraries
library(Seurat)
library(tidyverse)
library(ggplot2)
library(patchwork)

# Load the integrated object
output_dir <- "analysis/results"
figure_dir <- "analysis/figures"
seurat_integrated <- readRDS(file.path(output_dir, "integrated_seurat_object.rds"))

# Define cell type markers
cell_type_markers <- list(
    "Excitatory Neurons" = c("Slc17a7", "Satb2", "Neurod6"),
    "Inhibitory Neurons" = c("Gad1", "Gad2", "Dlx1", "Dlx2"),
    "Astrocytes" = c("Aldh1l1", "Aqp4", "Gfap"),
    "Oligodendrocytes" = c("Mbp", "Plp1", "Cnp"),
    "OPCs" = c("Pdgfra", "Cspg4", "Sox10"),
    "Microglia" = c("Cx3cr1", "P2ry12", "C1qa"),
    "Endothelial" = c("Cldn5", "Flt1", "Pecam1")
)

# Create feature plots for each cell type
for (cell_type in names(cell_type_markers)) {
    markers <- cell_type_markers[[cell_type]]
    
    # Generate feature plot
    feature_plot <- FeaturePlot(
        seurat_integrated,
        features = markers,
        ncol = 3,
        min.cutoff = "q10",
        max.cutoff = "q90"
    )
    
    # Save plot
    ggsave(
        file.path(figure_dir, paste0("feature_plot_", gsub(" ", "_", cell_type), ".pdf")),
        feature_plot,
        width = 12,
        height = 4 * ceiling(length(markers)/3)
    )
}

# Calculate module scores for each cell type
for (cell_type in names(cell_type_markers)) {
    seurat_integrated <- AddModuleScore(
        seurat_integrated,
        features = list(cell_type_markers[[cell_type]]),
        name = gsub(" ", "_", cell_type)
    )
}

# Generate module score heatmap
module_scores <- seurat_integrated[[]] %>%
    select(seurat_clusters, contains("_1")) %>%
    group_by(seurat_clusters) %>%
    summarise_all(mean) %>%
    column_to_rownames("seurat_clusters")

colnames(module_scores) <- names(cell_type_markers)

# Convert to matrix and create heatmap
module_scores_matrix <- as.matrix(module_scores)
heatmap_plot <- pheatmap::pheatmap(
    module_scores_matrix,
    color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
    main = "Cell Type Scores by Cluster",
    angle_col = 45,
    fontsize = 10
)

# Save heatmap
pdf(file.path(figure_dir, "cell_type_scores_heatmap.pdf"), width = 10, height = 8)
print(heatmap_plot)
dev.off()

# Assign cell types to clusters based on highest module scores
cluster_cell_types <- apply(module_scores_matrix, 1, function(x) {
    names(cell_type_markers)[which.max(x)]
})

# Add cell type annotations to the Seurat object
seurat_integrated$cell_type <- plyr::mapvalues(
    seurat_integrated$seurat_clusters,
    from = as.numeric(names(cluster_cell_types)),
    to = cluster_cell_types
)

# Generate UMAP plot with cell type annotations
cell_type_umap <- DimPlot(
    seurat_integrated,
    reduction = "umap",
    group.by = "cell_type",
    label = TRUE,
    repel = TRUE
) +
    ggtitle("Cell Types") +
    theme(legend.position = "right")

ggsave(file.path(figure_dir, "cell_type_umap.pdf"), cell_type_umap, width = 12, height = 8)

# Save annotated Seurat object
saveRDS(seurat_integrated, file.path(output_dir, "annotated_seurat_object.rds"))

# Generate cell type proportion plots
# By condition
cell_prop_condition <- seurat_integrated@meta.data %>%
    group_by(condition, cell_type) %>%
    summarise(count = n()) %>%
    group_by(condition) %>%
    mutate(proportion = count/sum(count))

prop_plot_condition <- ggplot(cell_prop_condition, 
                            aes(x = condition, y = proportion, fill = cell_type)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Cell Type Proportions by Condition",
         y = "Proportion",
         x = "Condition")

# By gene group
cell_prop_gene_group <- seurat_integrated@meta.data %>%
    group_by(gene_group, cell_type) %>%
    summarise(count = n()) %>%
    group_by(gene_group) %>%
    mutate(proportion = count/sum(count))

prop_plot_gene_group <- ggplot(cell_prop_gene_group, 
                              aes(x = gene_group, y = proportion, fill = cell_type)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Cell Type Proportions by Gene Group",
         y = "Proportion",
         x = "Gene Group")

# Combine and save proportion plots
combined_prop_plots <- prop_plot_condition + prop_plot_gene_group
ggsave(file.path(figure_dir, "cell_type_proportions.pdf"), 
       combined_prop_plots, 
       width = 15, 
       height = 7)

print("Cell type annotation complete. Results have been saved.")
print("Generated files:")
print("1. Feature plots for each cell type")
print("2. Cell type score heatmap")
print("3. Annotated UMAP plot")
print("4. Cell type proportion plots")
print("5. Annotated Seurat object") 