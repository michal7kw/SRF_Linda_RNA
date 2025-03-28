```python
# %% [markdown]
# # Environment
```


```python
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import anndata as ad
import warnings
from celltypist import models, annotate
```


```python
os.chdir("/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/cell_typist")
```


```python
# This cell will be parameterized by the script
sel_model = "Mouse_Isocortex_Hippocampus"  # This will be replaced with the actual model name
sel_sample = "Emx1_Mut"  # This will be replaced with the actual sample name
print(f"Processing model: {sel_model}, sample: {sel_sample}")

leiden_res_dict = {
    "Emx1_Ctrl": [0.38, 0.3, 0.05],
    "Emx1_Mut": [0.47, 0.38, 0.3],
    "Nestin_Ctrl": [0.05, 0.13, 0.47],
    "Nestin_Mut": [0.47, 0.3, 0.63]
}

leiden_res = leiden_res_dict[sel_sample]
```


```python
# Specific mouse brain models available in CellTypist
# https://www.celltypist.org/models
MOUSE_HIPPOCAMPUS_MODELS = {
    "Mouse_Isocortex_Hippocampus": {
        "description": "Cell types from the adult mouse isocortex (neocortex) and hippocampal formation",
        "cell_types": 42,
        "version": "v1",
        "reference": "https://doi.org/10.1016/j.cell.2021.04.021"
    },
    "Mouse_Dentate_Gyrus": {
        "description": "Cell types from the dentate gyrus in perinatal, juvenile, and adult mice",
        "cell_types": 24,
        "version": "v1",
        "reference": "https://doi.org/10.1038/s41593-017-0056-2"
    }
}
```


```python
# Set up directories
results_dir = f"results_{sel_model}"
model_dir = "models"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
```


```python
# DATA dirs
base_path = "/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis"
outputs_folder = "individual_data_analysis_opt_clusters"
folder_prefix = "cellranger_counts_R26_"
data_dir = os.path.join(base_path, outputs_folder, folder_prefix)

adata_paths = {
    "Emx1_Ctrl": f"{data_dir}Emx1_Ctrl_adult_0/Emx1_Ctrl_processed.h5ad",
    "Emx1_Mut": f"{data_dir}Emx1_Mut_adult_1/Emx1_Mut_processed.h5ad",
    "Nestin_Ctrl": f"{data_dir}Nestin_Ctrl_adult_2/Nestin_Ctrl_processed.h5ad",
    "Nestin_Mut": f"{data_dir}Nestin_Mut_adult_3/Nestin_Mut_processed.h5ad"
}
```


```python
adata_path = adata_paths[sel_sample]
model_path = f"models/{sel_model}.pkl"

# %% [markdown]
# # Load Data
```


```python
if adata_path:
    print(f"Loading AnnData from {adata_path}")
    adata = sc.read_h5ad(adata_path)
else:
    # Try to use a global adata object
    try:
        adata = globals()['adata']
        print("Using already loaded AnnData object")
    except KeyError:
        print("No AnnData object provided. Please provide a path to an .h5ad file.")

print(f"AnnData object contains {adata.n_obs} cells and {adata.n_vars} genes")
```


```python
model = models.Model.load(model_path)

# %% [markdown]
# # Explore cell annotation model
```


```python
print(type(model))
print(model.__dict__.keys())
print(model.description)
print(f"Model: {os.path.basename(model_path)}")
print(f"Number of cell types: {len(model.cell_types)}")
```


```python
# Inspect all available attributes and methods of the model object
print("Available attributes and methods:")
for attr in dir(model):
    if not attr.startswith('__'):  # Skip dunder methods
        attr_type = type(getattr(model, attr))
        print(f"  - {attr}: {attr_type}")
```


```python
hippo_suffix = ['CA1', 'CA2', 'CA3', 'DG', 'SUB-ProS']
cortical_suffix = ['CTX', 'L2', 'L3', 'L4', 'L5', 'L6']
```


```python
# Get all cell types
all_cell_types = model.cell_types

# Define hippocampal cell types
hippocampal_cell_types = [ct for ct in all_cell_types if any(h in ct for h in hippo_suffix)]
print("Hippocampal cell types:")
print(hippocampal_cell_types)

# Define cortical cell types (layers and cortical regions)
cortical_cell_types = [ct for ct in all_cell_types if any(c in ct for c in cortical_suffix)]
print("\nCortical cell types:")
print(cortical_cell_types)

# Other cell types
other_cell_types = [ct for ct in all_cell_types if ct not in hippocampal_cell_types and ct not in cortical_cell_types]
print("\nOther cell types:")
print(other_cell_types)

# Display original array for reference
model.cell_types
```


```python
print("\nCell types:")
for i, cell_type in enumerate(model.cell_types):
    print(f"  {i+1}. {cell_type}")
```


```python
# Extract some key marker genes
print("\nExtracting markers for key cell types...")
for cell_type in model.cell_types:
    markers = model.extract_top_markers(cell_type, 5)
    print(f"\nTop 5 markers for {cell_type}:")
    for marker in markers:
        print(f"  - {marker}")


# %% [markdown]
# # Annotate data
```


```python
non_zero_index = adata.raw.X[0].indices[0] if len(adata.raw.X[0].indices) > 0 else 0
print(adata.raw.X[0,12])
print(adata.X[0,12])
```


```python
adata.layers
```


```python
adata_norm = adata.copy()
```


```python
adata_norm.X = adata.layers['for_cell_typist']
```


```python
# Quick check that normalization worked correctly
counts_after_norm = np.expm1(adata_norm.X).sum(axis=1)
print(np.mean(counts_after_norm))

# Basic QC check
if np.mean(counts_after_norm) < 9000 or np.mean(counts_after_norm) > 11000:
    warnings.warn("Normalization may not have worked as expected. Check your data.")
```


```python
majority_voting = True
prob_threshold = 0.5
print(f"Running CellTypist with majority_voting={majority_voting}, prob_threshold={prob_threshold}")
predictions = annotate(
    adata_norm, 
    model=model_path,
    majority_voting=majority_voting,
    mode='prob match',  # Use probability-based matching for multi-label classification
    p_thres=prob_threshold
)
```


```python
# Add annotations to original adata
predictions.to_adata(adata_norm)
```


```python
# Also add probability scores for key cell types
predictions.to_adata(adata_norm, insert_prob=True, prefix='prob_')
```


```python
if 'X_umap' not in adata_norm.obsm:
    try:
        # Calculate neighborhood graph if not present
        if 'neighbors' not in adata_norm.uns:
            sc.pp.neighbors(adata_norm)
        sc.tl.umap(adata_norm)
    except Exception as e:
        print(f"Could not calculate UMAP: {e}")
        if 'X_pca' not in adata_norm.obsm:
            sc.pp.pca(adata_norm)

# %% [markdown]
# # Inspect results
```


```python
adata_norm.obs.columns
```


```python
# Cell type annotation plot
if 'majority_voting' in adata_norm.obs.columns:
    fig, ax = plt.subplots(figsize=(12, 10))
    sc.pl.umap(adata_norm, color='majority_voting', ax=ax, legend_loc='right margin', 
                title=f"Cell Type Annotation ({sel_model}, {sel_sample})")
    plt.tight_layout()
    output_file = os.path.join(results_dir, f"{sel_sample}_celltypes.png")
    fig.savefig(output_file, dpi=150)
    print(f"Saved cell type plot to {output_file}")
    plt.show()
    plt.close(fig)
```


```python
# Confidence score plot
if 'conf_score' in adata_norm.obs.columns:
    fig, ax = plt.subplots(figsize=(12, 10))
    sc.pl.umap(adata_norm, color='conf_score', ax=ax, 
                title=f"Annotation Confidence Score ({sel_model}, {sel_sample})", cmap='viridis')
    plt.tight_layout()
    output_file = os.path.join(results_dir, f"{sel_sample}_confidence.png")
    fig.savefig(output_file, dpi=150)
    print(f"Saved confidence score plot to {output_file}")
    plt.show()
    plt.close(fig)
```


```python
# Get probability columns
prob_columns = [col for col in adata_norm.obs.columns if col.startswith('prob_')]
```


```python
for i in range(0, len(prob_columns), 5):
    print("\t".join(prob_columns[i:i+5]))
```


```python
# Find hippocampal and cortical cell types
hippo_cols = [col for col in prob_columns if any(term in col for term in hippo_suffix)]
cortex_cols = [col for col in prob_columns if any(term in col for term in cortical_suffix)]
other_cols = [col for col in prob_columns if col not in hippo_cols and col not in cortex_cols]
```


```python
print(hippo_cols)
print(cortex_cols)
print(other_cols)

# %% [markdown]
# # Save data

output_file = os.path.join(results_dir, f"{sel_sample}_annotated.h5ad")
adata_norm.write(output_file)
print(f"Saved annotated data to {output_file}")
```


```python
# Create masks for different brain regions
hippo_mask = adata_norm.obs['majority_voting'].astype(str).str.contains('|'.join(hippo_suffix), case=False)
cortex_mask = adata_norm.obs['majority_voting'].astype(str).str.contains('|'.join(cortical_suffix), case=False)

# Create AnnData objects for each region
adata_regions = {}

if hippo_mask.sum() > 0:
    print(f"Extracting {hippo_mask.sum()} hippocampal cells")
    adata_regions['hippocampus'] = adata_norm[hippo_mask].copy()
    
    # Calculate UMAP if needed
    if 'X_umap' not in adata_regions['hippocampus'].obsm:
        try:
            sc.pp.neighbors(adata_regions['hippocampus'])
            sc.tl.umap(adata_regions['hippocampus'])
        except:
            print("Could not calculate UMAP for hippocampal cells")

if cortex_mask.sum() > 0:
    print(f"Extracting {cortex_mask.sum()} cortical cells")
    adata_regions['cortex'] = adata_norm[cortex_mask].copy()
    
    # Calculate UMAP if needed
    if 'X_umap' not in adata_regions['cortex'].obsm:
        try:
            sc.pp.neighbors(adata_regions['cortex'])
            sc.tl.umap(adata_regions['cortex'])
        except:
            print("Could not calculate UMAP for cortical cells")

# Create a mask for the remaining cells
other_mask = ~(hippo_mask | cortex_mask)

if other_mask.sum() > 0:
    print(f"Extracting {other_mask.sum()} other cells")
    adata_regions['other'] = adata_norm[other_mask].copy()
    
    # Calculate UMAP if needed
    if 'X_umap' not in adata_regions['other'].obsm:
        try:
            sc.pp.neighbors(adata_regions['other'])
            sc.tl.umap(adata_regions['other'])
        except:
            print("Could not calculate UMAP for other cells")
```


```python
# Save region-specific data
for region, adata_region in adata_regions.items():
    output_path = os.path.join(results_dir, f"{region}_cells.h5ad")
    adata_region.write(output_path)
    print(f"{region.capitalize()} cells saved to {output_path}")
```


```python
# Generate a summary table of cell type annotations
cell_type_counts = adata_norm.obs['majority_voting'].value_counts()
cell_type_df = pd.DataFrame({
    'cell_type': cell_type_counts.index,
    'cell_count': cell_type_counts.values,
    'percentage': (cell_type_counts.values / cell_type_counts.sum() * 100).round(2)
})
cell_type_df = cell_type_df.sort_values('cell_count', ascending=False).reset_index(drop=True)

print(f"\nSummary of cell types for {sel_sample} using {sel_model} model:")
display(cell_type_df)
```


```python
# Save summary to CSV
summary_file = os.path.join(results_dir, f"{sel_sample}_cell_type_summary.csv")
cell_type_df.to_csv(summary_file, index=False)
print(f"Saved cell type summary to {summary_file}")
```


```python
print(f"\n{'='*50}")
print(f"CELLTYPIST ANALYSIS COMPLETED")
print(f"{'='*50}")
print(f"Sample: {sel_sample}")
print(f"Model: {sel_model}")
print(f"Number of cells: {adata_norm.n_obs}")
print(f"Number of cell types identified: {len(cell_type_counts)}")
print(f"Results saved to: {os.path.abspath(results_dir)}")
print(f"{'='*50}")



```
