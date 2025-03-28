# %%
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import anndata as ad
import requests
import urllib.request
import warnings

WORKING_DIR = "/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/check_markers"
os.chdir(WORKING_DIR)
sys.path.append(WORKING_DIR)

from functions import *

# This will be added by the parameterized notebook script:
# OUTPUT_DIR = os.path.join(WORKING_DIR, "results", "MODEL_TYPE", "SAMPLE_PLACEHOLDER")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# # Define gene sets

# %%
gene_list = pd.read_csv("FirstLevelGeneList.csv")
gene_list

# %% [markdown]
# # Load data

# %%
# DATA dirs

# This cell will be parameterized by the script
SAMPLE_NAME = "SAMPLE_PLACEHOLDER"  # This will be replaced with the actual sample name
# SAMPLE_NAME = "Emx1_Ctrl"
print(f"Processing sample: {SAMPLE_NAME}")

# This cell will be parameterized by the script
MODEL_TYPE = "MODEL_TYPE"  # This will be replaced with the actual model type
# MODEL_TYPE = "Dentate_Gyrus"
print(f"Processing model: {MODEL_TYPE}")

# %%
data_path = f"/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/cell_typist/results_{MODEL_TYPE}"

adata_paths = {
    "Emx1_Ctrl": f"{data_path}/Emx1_Ctrl_annotated.h5ad",
    "Emx1_Mut": f"{data_path}/Emx1_Mut_annotated.h5ad",
    "Nestin_Ctrl": f"{data_path}/Nestin_Ctrl_annotated.h5ad",
    "Nestin_Mut": f"{data_path}/Nestin_Mut_annotated.h5ad"
}

# Load AnnData objects into a dictionary
# adata_dict = {}
# for key, path in adata_paths.items():
#     print(f"Loading AnnData from {path}")
#     adata_dict[key] = sc.read_h5ad(path)
#     print(f"AnnData object {key} contains {adata_dict[key].n_obs} cells and {adata_dict[key].n_vars} genes")

# %%
adata = sc.read_h5ad(adata_paths[SAMPLE_NAME])

# %%
adata

# %% [markdown]
# # Check Biomarkers

# %%
with pd.option_context("display.max_columns", None):
    adata.obs.head()

# %%
# Save the UMAP plot to the output directory
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# prob_conf_score - use a sequential colormap (Reds)
sc.pl.umap(adata, color='prob_conf_score', ax=axs[0], show=False, cmap='Reds', title='Confidence Score')

# leiden_0.38 - use a categorical palette
sc.pl.umap(adata, color='leiden_0.38', ax=axs[1], show=False, palette='tab20', title='Leiden Clusters')

# majority_voting - use a categorical palette
sc.pl.umap(adata, color='majority_voting', ax=axs[2], show=False, palette='tab20', title='Cell Types')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{SAMPLE_NAME}_umap.png"), dpi=300, bbox_inches='tight')
plt.close()

# Display for notebook - individual plots with proper coloring
sc.pl.umap(adata, color='prob_conf_score', cmap='Reds', title='Confidence Score')
sc.pl.umap(adata, color='leiden_0.38', palette='tab20', title='Leiden Clusters')
sc.pl.umap(adata, color='majority_voting', palette='tab20', title='Cell Types')

# %%
cell_types = gene_list.columns.tolist()
print(cell_types)

# %%
markers_dict = {col: gene_list[col].dropna().tolist() for col in gene_list.columns}
markers_dict

# %%
# Update plot_marker_genes function call to save plots to the output directory
def plot_marker_genes_with_save(adata, cell_type, markers_dict, output_dir):
    """Modified plotting function that saves plots to specified directory"""
    markers = markers_dict[cell_type]
    
    # Check which markers are in the dataset
    available_markers = [m for m in markers if m in adata.var_names]
    
    if not available_markers:
        print(f"Warning: None of the markers for {cell_type} are in the dataset")
        return
    
    # Replace slashes in cell type name with hyphens for filenames
    safe_cell_type = cell_type.replace('/', '-')
    
    # Calculate grid layout
    n_markers = len(available_markers)
    n_cols = min(3, n_markers)  # Max 3 columns
    n_rows = (n_markers + n_cols - 1) // n_cols  # Ceiling division
    
    # Create a single figure for all markers of this cell type
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    
    for i, marker in enumerate(available_markers):
        print(f"Plotting {marker} for {cell_type}")
        ax = plt.subplot(n_rows, n_cols, i + 1)
        sc.pl.umap(adata, color=marker, title=f"{marker}", 
                  cmap='Reds', show=False, ax=ax)
    
    # Add a main title for the entire figure
    plt.suptitle(f"Marker genes for {cell_type}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust to make room for suptitle
    
    # Save the combined figure
    plt.savefig(os.path.join(output_dir, f"{safe_cell_type}_markers_combined.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also display in notebook - this uses scanpy's built-in multi-panel plot
    print(f"Showing combined plot for {cell_type} markers:")
    sc.pl.umap(adata, color=available_markers, ncols=n_cols, cmap='Reds')

# %%
for selected_cell_type in markers_dict.keys():
    print(f"Processing {selected_cell_type}")
    plot_marker_genes_with_save(adata, selected_cell_type, markers_dict, OUTPUT_DIR)
    
    # Also save summary plot for this cell type 
    available_markers = [m for m in markers_dict[selected_cell_type] if m in adata.var_names]
    if available_markers:
        # Replace slashes in cell type name with hyphens for filenames
        safe_cell_type = selected_cell_type.replace('/', '-')
        
        # Get a dotplot of all markers for this cell type
        plt.figure(figsize=(12, 6))
        sc.pl.dotplot(adata, available_markers, groupby='majority_voting', 
                      title=f"{selected_cell_type} markers", show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_cell_type}_dotplot.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

# %%
# Save the final results as an annotated h5ad file
adata.write_h5ad(os.path.join(OUTPUT_DIR, f"{SAMPLE_NAME}_analyzed.h5ad"))

# Generate a simple summary report
with open(os.path.join(OUTPUT_DIR, f"{SAMPLE_NAME}_summary.txt"), 'w') as f:
    f.write(f"Analysis summary for {SAMPLE_NAME} with model {MODEL_TYPE}\n")
    f.write(f"Total cells: {adata.n_obs}\n")
    f.write(f"Total genes: {adata.n_vars}\n")
    f.write("\nCell type distribution:\n")
    cell_type_counts = adata.obs['majority_voting'].value_counts()
    for cell_type, count in cell_type_counts.items():
        f.write(f"{cell_type}: {count} cells ({count/adata.n_obs:.2%})\n")

print(f"Analysis complete for {SAMPLE_NAME}. Results saved to {OUTPUT_DIR}")



