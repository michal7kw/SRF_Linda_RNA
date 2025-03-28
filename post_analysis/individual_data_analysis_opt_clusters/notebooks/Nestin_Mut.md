```python
# Import libraries
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.sparse import issparse
from datetime import datetime
from tqdm import tqdm
from sklearn import metrics
from scipy import stats
import warnings
import sys


# Set plotting settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, frameon=False)

BASE_DIR = "/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/"
```


```python
import warnings
warnings.filterwarnings("ignore")
```


```python
samples = {
    "Emx1_Ctrl": "cellranger_counts_R26_Emx1_Ctrl_adult_0",
    "Emx1_Mut": "cellranger_counts_R26_Emx1_Mut_adult_1",
    "Nestin_Ctrl": "cellranger_counts_R26_Nestin_Ctrl_adult_2",
    "Nestin_Mut": "cellranger_counts_R26_Nestin_Mut_adult_3"
    }
```


```python
# This cell will be parameterized by the script
SAMPLE_NAME = "Nestin_Mut"  # This will be replaced with the actual sample name
# SAMPLE_NAME = "Emx1_Ctrl"
print(f"Processing sample: {SAMPLE_NAME}")

# %% [markdown]
# # 1. Setup and Data Loading
```

    Processing sample: Nestin_Mut



```python
SAMPLE = samples[SAMPLE_NAME]

WORKING_DIR = os.path.join(BASE_DIR, "post_analysis", "individual_data_analysis_opt_clusters", SAMPLE)
os.makedirs(WORKING_DIR, exist_ok=True)

CELL_DATA_DIR = "cellranger_final_count_data"
matrix_dir = os.path.join(BASE_DIR, CELL_DATA_DIR, SAMPLE, "outs", "filtered_feature_bc_matrix")

os.chdir(WORKING_DIR)
OUTPUT_DIR=WORKING_DIR

sys.path.append(os.path.join(BASE_DIR, "post_analysis", "individual_data_analysis_opt_clusters"))
from functions import *

# Load the data from the filtered matrix
try:
    adata = sc.read_10x_mtx(
        matrix_dir,
        var_names='gene_symbols',
        cache=True
    )
    print(f"Shape of loaded data: {adata.shape}")  # cells × genes
except ValueError as e:
    print(f"Error loading data: {e}")
    # Try loading with different parameters to handle the mismatch
    adata = sc.read_10x_mtx(
        matrix_dir,
        var_names='gene_symbols',
        cache=False
    )
    print(f"Shape of loaded data after retry: {adata.shape}")  # cells × genes

# %% [markdown]
# # 2. Basic Pre-processing
```

    ... reading from cache file cache/beegfs-scratch-ric.broccoli-kubacki.michal-SRF_Linda_RNA-cellranger_final_count_data-cellranger_counts_R26_Nestin_Mut_adult_3-outs-filtered_feature_bc_matrix-matrix.h5ad


    Shape of loaded data: (9423, 33696)



```python
# Make a copy of the raw counts
adata.raw = adata.copy()

# Basic filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Calculate QC metrics
adata.var['mt'] = adata.var_names.str.startswith('mt-')  # identify mitochondrial genes
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# Plot QC metrics
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(adata.obs['n_genes_by_counts'], kde=False, ax=axs[0])
axs[0].set_title('Genes per cell')
sns.histplot(adata.obs['total_counts'], kde=False, ax=axs[1])
axs[1].set_title('UMI counts per cell')
sns.histplot(adata.obs['pct_counts_mt'], kde=False, ax=axs[2])
axs[2].set_title('Percent mitochondrial')
plt.tight_layout()

# Save the plot to the output directory
plt.savefig(os.path.join(OUTPUT_DIR, 'qc_metrics.png'))
plt.show()

# %% [markdown]
# # 3. Filtering Based on QC Metrics
```

    filtered out 7601 genes that are detected in less than 3 cells



    
![png](Nestin_Mut_files/Nestin_Mut_5_1.png)
    



```python
max_genes = 15000 
min_genes = 500  
max_mt_pct = 20  

adata = adata[adata.obs['n_genes_by_counts'] < max_genes, :]
adata = adata[adata.obs['n_genes_by_counts'] > min_genes, :]
adata = adata[adata.obs['pct_counts_mt'] < max_mt_pct, :]

print(f"Number of cells after filtering: {adata.n_obs}")
print(f"Number of genes after filtering: {adata.n_vars}")

# %% [markdown]
# # 4. Normalization and Log Transformation
```

    Number of cells after filtering: 9237
    Number of genes after filtering: 26095



```python
# Normalize to 10,000 reads per cell
sc.pp.normalize_total(adata, target_sum=1e4)

# Log transform
sc.pp.log1p(adata)

# Identify highly variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
print(f"Number of highly variable genes: {sum(adata.var.highly_variable)}")

# Plot highly variable genes
plt.figure(figsize=(10, 8))
sc.pl.highly_variable_genes(adata, show=False)
plt.tight_layout()
plt.show()
```

    normalizing counts per cell


        finished (0:00:00)


    extracting highly variable genes


        finished (0:00:00)


    --> added
        'highly_variable', boolean vector (adata.var)
        'means', float vector (adata.var)
        'dispersions', float vector (adata.var)
        'dispersions_norm', float vector (adata.var)


    Number of highly variable genes: 5001



    <Figure size 1000x800 with 0 Axes>



    
![png](Nestin_Mut_files/Nestin_Mut_7_7.png)
    



```python
# Save the current normalized and log-transformed data to a new layer BEFORE scaling
adata.layers['for_cell_typist'] = adata.X.copy()
```


```python
# Quick check that the data in the layer is correctly normalized
# Reverse log1p transformation
if issparse(adata.layers['for_cell_typist']):
    counts_in_layer = adata.layers['for_cell_typist'].copy()
    counts_in_layer.data = np.expm1(counts_in_layer.data)
else:
    counts_in_layer = np.expm1(adata.layers['for_cell_typist'])

# Sum counts per cell
total_counts_layer = np.asarray(counts_in_layer.sum(axis=1)).flatten()

print("\nVerifying normalization in 'for_cell_typist' layer:")
print(f"  Mean total counts (reversed log1p): {total_counts_layer.mean():.2f}")
print(f"  Median total counts (reversed log1p): {np.median(total_counts_layer):.2f}")

# Basic QC check for the layer
if np.mean(total_counts_layer) < 9900 or np.mean(total_counts_layer) > 10100:
    warnings.warn(f"Normalization in 'for_cell_typist' layer may not be exactly 10k (Mean: {total_counts_layer.mean():.2f}). Check normalization step.")
else:
    print("  Normalization in 'for_cell_typist' layer appears correct (around 10k).")

# %% [markdown]
# # 5. Dimensionality Reduction
```

    
    Verifying normalization in 'for_cell_typist' layer:
      Mean total counts (reversed log1p): 10000.00
      Median total counts (reversed log1p): 10000.00
      Normalization in 'for_cell_typist' layer appears correct (around 10k).



```python
# Scale adata.X to unit variance and zero mean AFTER saving the normalized layer
# This step modifies adata.X but leaves adata.layers['for_cell_typist'] untouched
sc.pp.scale(adata, max_value=10)

# Run PCA
sc.tl.pca(adata, svd_solver='arpack')

# Determine number of significant PCs
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)
plt.show()
```

    ... as `zero_center=True`, sparse input is densified and may lead to large memory consumption


    computing PCA


        with n_comps=50


        finished (0:00:15)



    
![png](Nestin_Mut_files/Nestin_Mut_10_4.png)
    



```python
# Choose number of PCs for downstream analyses
n_pcs = 30  # Adjust based on the variance ratio plot

# Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)

# Run UMAP
sc.tl.umap(adata)

# Plot UMAP
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], 
        use_raw=False, color_map='viridis', show=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# # 6. Marker Gene Identification
```

    computing neighbors


        using 'X_pca' with n_pcs = 30


        finished: added to `.uns['neighbors']`
        `.obsp['distances']`, distances for each pair of neighbors
        `.obsp['connectivities']`, weighted adjacency matrix (0:00:23)


    computing UMAP


        finished: added
        'X_umap', UMAP coordinates (adata.obsm)
        'umap', UMAP parameters (adata.uns) (0:00:17)



    <Figure size 1000x800 with 0 Axes>



    
![png](Nestin_Mut_files/Nestin_Mut_11_6.png)
    



```python
# Try different resolutions to find optimal number of clusters
resolution_range=[0.05, 0.8]
n_resolutions=10
resolutions = np.linspace(resolution_range[0], resolution_range[1], n_resolutions)
resolutions = [round(r, 2) for r in resolutions]
```


```python
# Check first 5 values from first cell
if issparse(adata.X):
    print("X matrix values (first cell):", adata.X[0, :5].toarray().flatten())
else:
    print("X matrix values (first cell):", adata.X[0, :5])
print("Should be log1p transformed values (~0-5 range)")

# Check raw values if raw exists
if adata.raw:
    if issparse(adata.raw.X):
        print("Raw values:", adata.raw.X[0, :5].toarray().flatten())
    else:
        print("Raw values:", adata.raw.X[0, :5])
    print("Should be original counts (integers)")

```

    X matrix values (first cell): [ 0.96394587 -0.2894199   2.056085   -0.02714088 -0.09086135]
    Should be log1p transformed values (~0-5 range)
    Raw values: [16.  0.  1.  0.  0.]
    Should be original counts (integers)



```python
# With custom parameters
optimal_resolution = analyze_and_select_best_clustering(
    adata,
    resolutions=resolutions,
    run_marker_analysis=True,       # Run marker gene analysis
    leiden_key='leiden',            # Base name for cluster labels
    output_dir="my_cluster_analysis"  # Output directory
)

# Annotate adata with optimal clustering (if not already present)
best_clustering = f"leiden_{optimal_resolution}"
if best_clustering not in adata.obs:
    sc.tl.leiden(adata, resolution=optimal_resolution, key_added=best_clustering)
```

    Analyzing 10 clustering resolutions: [0.05, 0.13, 0.22, 0.3, 0.38, 0.47, 0.55, 0.63, 0.72, 0.8]
    
    Step 1: Running Leiden clustering at different resolutions...


    Computing clusterings:   0%|          | 0/10 [00:00<?, ?it/s]

    running Leiden clustering


        finished: found 7 clusters and added
        'leiden_0.05', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  10%|█         | 1/10 [00:00<00:03,  2.92it/s]

    running Leiden clustering


        finished: found 12 clusters and added
        'leiden_0.13', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  20%|██        | 2/10 [00:00<00:02,  2.95it/s]

    running Leiden clustering


        finished: found 14 clusters and added
        'leiden_0.22', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  30%|███       | 3/10 [00:01<00:02,  2.95it/s]

    running Leiden clustering


        finished: found 18 clusters and added
        'leiden_0.3', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  40%|████      | 4/10 [00:01<00:02,  2.96it/s]

    running Leiden clustering


        finished: found 19 clusters and added
        'leiden_0.38', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  50%|█████     | 5/10 [00:01<00:01,  2.56it/s]

    running Leiden clustering


        finished: found 23 clusters and added
        'leiden_0.47', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  60%|██████    | 6/10 [00:02<00:02,  1.96it/s]

    running Leiden clustering


        finished: found 25 clusters and added
        'leiden_0.55', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  70%|███████   | 7/10 [00:03<00:01,  1.95it/s]

    running Leiden clustering


        finished: found 25 clusters and added
        'leiden_0.63', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  80%|████████  | 8/10 [00:03<00:01,  1.85it/s]

    running Leiden clustering


        finished: found 26 clusters and added
        'leiden_0.72', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  90%|█████████ | 9/10 [00:04<00:00,  1.87it/s]

    running Leiden clustering


        finished: found 30 clusters and added
        'leiden_0.8', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings: 100%|██████████| 10/10 [00:04<00:00,  1.77it/s]

    Computing clusterings: 100%|██████████| 10/10 [00:04<00:00,  2.06it/s]

    


    
    Step 2: Identifying marker genes for each clustering resolution...


    Processing resolutions:   0%|          | 0/10 [00:00<?, ?it/s]

    
    Analyzing resolution 0.05:
    ranking genes


        finished: added to `.uns['rank_genes_0.05']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:30)


      ✓ Identified differentially expressed genes


    Processing resolutions:  10%|█         | 1/10 [00:32<04:50, 32.33s/it]

      ✓ Generated marker ranking plot
      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.05.csv
    
    Analyzing resolution 0.13:
    ranking genes


        finished: added to `.uns['rank_genes_0.13']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:44)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  20%|██        | 2/10 [01:20<05:31, 41.48s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.13.csv
    
    Analyzing resolution 0.22:
    ranking genes


        finished: added to `.uns['rank_genes_0.22']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:50)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  30%|███       | 3/10 [02:14<05:32, 47.49s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.22.csv
    
    Analyzing resolution 0.3:
    ranking genes


        finished: added to `.uns['rank_genes_0.3']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:02)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  40%|████      | 4/10 [03:21<05:31, 55.24s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.3.csv
    
    Analyzing resolution 0.38:
    ranking genes


        finished: added to `.uns['rank_genes_0.38']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:04)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  50%|█████     | 5/10 [04:31<05:01, 60.26s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.38.csv
    
    Analyzing resolution 0.47:
    ranking genes


        finished: added to `.uns['rank_genes_0.47']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:16)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  60%|██████    | 6/10 [05:52<04:29, 67.41s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.47.csv
    
    Analyzing resolution 0.55:
    ranking genes


        finished: added to `.uns['rank_genes_0.55']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:18)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  70%|███████   | 7/10 [07:17<03:39, 73.11s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.55.csv
    
    Analyzing resolution 0.63:
    ranking genes


        finished: added to `.uns['rank_genes_0.63']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:14)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  80%|████████  | 8/10 [08:36<02:30, 75.16s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.63.csv
    
    Analyzing resolution 0.72:
    ranking genes


        finished: added to `.uns['rank_genes_0.72']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:15)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  90%|█████████ | 9/10 [09:58<01:17, 77.14s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.72.csv
    
    Analyzing resolution 0.8:
    ranking genes


        finished: added to `.uns['rank_genes_0.8']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:25)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions: 100%|██████████| 10/10 [11:30<00:00, 81.82s/it]

    Processing resolutions: 100%|██████████| 10/10 [11:30<00:00, 69.06s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.8.csv
    
    Summary comparison saved to my_cluster_analysis/marker_analysis/resolution_comparison_summary.csv


    


    
    Analysis complete. Results saved to my_cluster_analysis/marker_analysis/
    
    Step 3: Evaluating clustering quality and selecting optimal resolution...
    Evaluating clustering metrics across resolutions...


    Evaluating resolutions:   0%|          | 0/10 [00:00<?, ?it/s]

    Evaluating resolutions:  10%|█         | 1/10 [00:00<00:06,  1.36it/s]

    Evaluating resolutions:  20%|██        | 2/10 [00:01<00:05,  1.43it/s]

    Evaluating resolutions:  30%|███       | 3/10 [00:02<00:04,  1.45it/s]

    Evaluating resolutions:  40%|████      | 4/10 [00:02<00:04,  1.47it/s]

    Evaluating resolutions:  50%|█████     | 5/10 [00:03<00:03,  1.48it/s]

    Evaluating resolutions:  60%|██████    | 6/10 [00:04<00:02,  1.49it/s]

    Evaluating resolutions:  70%|███████   | 7/10 [00:04<00:01,  1.50it/s]

    Evaluating resolutions:  80%|████████  | 8/10 [00:05<00:01,  1.51it/s]

    Evaluating resolutions:  90%|█████████ | 9/10 [00:06<00:00,  1.51it/s]

    Evaluating resolutions: 100%|██████████| 10/10 [00:06<00:00,  1.51it/s]

    Evaluating resolutions: 100%|██████████| 10/10 [00:06<00:00,  1.49it/s]

    


    
    Optimal clustering resolution: 0.47
    Optimal number of clusters: 23
    Metrics saved to my_cluster_analysis/evaluation/clustering_quality_metrics.csv


    Detailed metric analysis saved to my_cluster_analysis/evaluation/metric_details
    
    Analysis complete in 708.1 seconds!
    Optimal resolution: 0.47 (23 clusters)
    All clustering resolutions have been preserved in the AnnData object
    Full results saved to /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Nestin_Mut_adult_3/my_cluster_analysis



    <Figure size 1500x1000 with 0 Axes>



    <Figure size 1500x1000 with 0 Axes>



    <Figure size 1500x1000 with 0 Axes>



    <Figure size 1500x1000 with 0 Axes>



    <Figure size 1500x1000 with 0 Axes>



    <Figure size 1500x1000 with 0 Axes>



    <Figure size 1500x1000 with 0 Axes>



    <Figure size 1500x1000 with 0 Axes>



    <Figure size 1500x1000 with 0 Axes>



    <Figure size 1500x1000 with 0 Axes>



    <Figure size 1000x800 with 0 Axes>



```python
# Load the CSV file
df = pd.read_csv(os.path.join(OUTPUT_DIR, 'my_cluster_analysis', 'evaluation', 'clustering_quality_metrics.csv'))

# Sort the dataframe by overall_score in descending order
sorted_df = df.sort_values(by='overall_score', ascending=False)

# Create an ordered list of resolutions
ordered_resolutions = sorted_df['resolution'].tolist()
scores = []
print("Resolutions ordered by overall_score (highest to lowest):")
for i, res in enumerate(ordered_resolutions, 1):
    score = sorted_df.loc[sorted_df['resolution'] == res, 'overall_score'].values[0]
    scores.append(score)
    print(f"{i}. Resolution: {res}, Overall Score: {score}")
```

    Resolutions ordered by overall_score (highest to lowest):
    1. Resolution: 0.47, Overall Score: 0.7624248560892509
    2. Resolution: 0.3, Overall Score: 0.7579350037952377
    3. Resolution: 0.63, Overall Score: 0.7312750377388112
    4. Resolution: 0.55, Overall Score: 0.6982719702713367
    5. Resolution: 0.38, Overall Score: 0.6937747265195977
    6. Resolution: 0.22, Overall Score: 0.654093634050023
    7. Resolution: 0.13, Overall Score: 0.5524117881658356
    8. Resolution: 0.8, Overall Score: 0.4514430897885822
    9. Resolution: 0.05, Overall Score: 0.45
    10. Resolution: 0.72, Overall Score: 0.3824038464750352



```python
# Try different resolutions to find optimal number of clusters
best_resolutions = ordered_resolutions[:3]
print(best_resolutions)
# Plot clusters at different resolutions with improved layout
fig, axes = plt.subplots(1, len(best_resolutions), figsize=(20, 5))
for i, res in enumerate(best_resolutions):
    sc.pl.umap(adata, color=f'leiden_{res}', title=f'Resolution {res}, score {scores[i]}', 
               frameon=True, legend_loc='on data', legend_fontsize=10, ax=axes[i], show=False)

# Ensure proper spacing between subplots
plt.tight_layout()
plt.show()

# %% [markdown]
# # 7. Save Processed Data
```

    [0.47, 0.3, 0.63]



    
![png](Nestin_Mut_files/Nestin_Mut_16_1.png)
    



```python
# Define the output file path
output_adata_file = os.path.join(OUTPUT_DIR, f"{SAMPLE_NAME}_processed.h5ad")

# List all clustering assignments stored in the adata object
print("Clustering assignments stored in the AnnData object:")
leiden_columns = [col for col in adata.obs.columns if col.startswith('leiden_')]
for col in leiden_columns:
    n_clusters = len(adata.obs[col].unique())
    print(f"  - {col}: {n_clusters} clusters")

# Save the AnnData object with all clustering results
print(f"\nSaving processed AnnData object to: {output_adata_file}")
try:
    adata.write(output_adata_file)
    print("Successfully saved AnnData object with all clustering assignments.")
except Exception as e:
    print(f"Error saving AnnData object: {e}")

# %% [markdown]
# # 8. Visualize Clustering Results and Quality Metrics
```

    Clustering assignments stored in the AnnData object:
      - leiden_0.05: 7 clusters
      - leiden_0.13: 12 clusters
      - leiden_0.22: 14 clusters
      - leiden_0.3: 18 clusters
      - leiden_0.38: 19 clusters
      - leiden_0.47: 23 clusters
      - leiden_0.55: 25 clusters
      - leiden_0.63: 25 clusters
      - leiden_0.72: 26 clusters
      - leiden_0.8: 30 clusters
    
    Saving processed AnnData object to: /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Nestin_Mut_adult_3/Nestin_Mut_processed.h5ad


    Successfully saved AnnData object with all clustering assignments.



```python
# Display the optimal clustering on UMAP
plt.figure(figsize=(12, 10))
sc.pl.umap(adata, color=f'leiden_{optimal_resolution}', 
           title=f'Optimal Clustering (Resolution={optimal_resolution})', 
           legend_loc='on data', frameon=True, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'optimal_clustering_umap.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 8.1 Clustering Quality Metrics Analysis
```


    <Figure size 1200x1000 with 0 Axes>



    
![png](Nestin_Mut_files/Nestin_Mut_18_1.png)
    



```python
# Load the clustering quality metrics
metrics_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'my_cluster_analysis', 'evaluation', 'clustering_quality_metrics.csv'))
print("Clustering quality metrics summary:")
display(metrics_df[['resolution', 'n_clusters', 'silhouette_score', 'davies_bouldin_score', 'marker_gene_score', 'overall_score']])
```

    Clustering quality metrics summary:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>resolution</th>
      <th>n_clusters</th>
      <th>silhouette_score</th>
      <th>davies_bouldin_score</th>
      <th>marker_gene_score</th>
      <th>overall_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.05</td>
      <td>7</td>
      <td>0.053825</td>
      <td>-1.792387</td>
      <td>1.0</td>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.13</td>
      <td>12</td>
      <td>0.079772</td>
      <td>-1.982148</td>
      <td>1.0</td>
      <td>0.552412</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.22</td>
      <td>14</td>
      <td>0.115886</td>
      <td>-1.937675</td>
      <td>1.0</td>
      <td>0.654094</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.30</td>
      <td>18</td>
      <td>0.111732</td>
      <td>-1.827834</td>
      <td>1.0</td>
      <td>0.757935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.38</td>
      <td>19</td>
      <td>0.077426</td>
      <td>-1.908792</td>
      <td>1.0</td>
      <td>0.693775</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.47</td>
      <td>23</td>
      <td>0.093280</td>
      <td>-1.821234</td>
      <td>1.0</td>
      <td>0.762425</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.55</td>
      <td>25</td>
      <td>0.091972</td>
      <td>-1.958772</td>
      <td>1.0</td>
      <td>0.698272</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.63</td>
      <td>25</td>
      <td>0.098316</td>
      <td>-1.973544</td>
      <td>1.0</td>
      <td>0.731275</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.72</td>
      <td>26</td>
      <td>0.106312</td>
      <td>-1.923534</td>
      <td>0.5</td>
      <td>0.382404</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.80</td>
      <td>30</td>
      <td>0.105149</td>
      <td>-1.917823</td>
      <td>0.5</td>
      <td>0.451443</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Display the main clustering quality metrics visualization
from IPython.display import Image, display

print("Clustering quality metrics across resolutions:")
metrics_img = Image(os.path.join(OUTPUT_DIR, 'my_cluster_analysis', 'evaluation', 'clustering_quality_metrics.png'))
display(metrics_img)
```

    Clustering quality metrics across resolutions:



    
![png](Nestin_Mut_files/Nestin_Mut_20_1.png)
    



```python
# Display metric contributions visualization if available
metric_details_path = os.path.join(OUTPUT_DIR, 'my_cluster_analysis', 'evaluation', 'metric_details')
if os.path.exists(metric_details_path):
    contributions_img = Image(os.path.join(metric_details_path, 'metric_contributions.png'))
    print("Contribution of each metric to the overall score:")
    display(contributions_img)
    
    individual_metrics_img = Image(os.path.join(metric_details_path, 'individual_metrics.png'))
    print("Individual metrics across resolutions:")
    display(individual_metrics_img)
```

    Contribution of each metric to the overall score:



    
![png](Nestin_Mut_files/Nestin_Mut_21_1.png)
    


    Individual metrics across resolutions:



    
![png](Nestin_Mut_files/Nestin_Mut_21_3.png)
    



```python
# Load and display the metric contribution summary
contribution_summary_path = os.path.join(OUTPUT_DIR, 'my_cluster_analysis', 'evaluation', 'metric_details', 'metric_contribution_summary.csv')
if os.path.exists(contribution_summary_path):
    contribution_df = pd.read_csv(contribution_summary_path)
    print("Metric contribution summary:")
    display(contribution_df)

# %% [markdown]
# ## 8.2 Marker Genes for Optimal Clustering
```

    Metric contribution summary:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>resolution</th>
      <th>n_clusters</th>
      <th>overall_score</th>
      <th>silhouette_score_normalized_contribution</th>
      <th>davies_bouldin_score_normalized_contribution</th>
      <th>calinski_harabasz_score_normalized_contribution</th>
      <th>avg_marker_genes_normalized_contribution</th>
      <th>marker_gene_score_normalized_contribution</th>
      <th>marker_gene_significance_normalized_contribution</th>
      <th>cluster_separation_score_normalized_contribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.05</td>
      <td>7</td>
      <td>0.450000</td>
      <td>0.000000</td>
      <td>0.220000</td>
      <td>0.220000</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.13</td>
      <td>12</td>
      <td>0.552412</td>
      <td>0.146331</td>
      <td>0.000000</td>
      <td>0.118945</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.055103</td>
      <td>0.011969</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.22</td>
      <td>14</td>
      <td>0.654094</td>
      <td>0.350000</td>
      <td>0.051560</td>
      <td>0.117842</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.055290</td>
      <td>0.009717</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.30</td>
      <td>18</td>
      <td>0.757935</td>
      <td>0.326571</td>
      <td>0.178904</td>
      <td>0.068733</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.067111</td>
      <td>0.021909</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.38</td>
      <td>19</td>
      <td>0.693775</td>
      <td>0.133099</td>
      <td>0.085046</td>
      <td>0.045646</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.080000</td>
      <td>0.031832</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.47</td>
      <td>23</td>
      <td>0.762425</td>
      <td>0.222510</td>
      <td>0.186556</td>
      <td>0.023161</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.063718</td>
      <td>0.040611</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.55</td>
      <td>25</td>
      <td>0.698272</td>
      <td>0.215137</td>
      <td>0.027101</td>
      <td>0.013037</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.061547</td>
      <td>0.045990</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.63</td>
      <td>25</td>
      <td>0.731275</td>
      <td>0.250915</td>
      <td>0.009976</td>
      <td>0.016164</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.072421</td>
      <td>0.045202</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.72</td>
      <td>26</td>
      <td>0.382404</td>
      <td>0.296008</td>
      <td>0.067955</td>
      <td>0.018981</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.021753</td>
      <td>0.040411</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.80</td>
      <td>30</td>
      <td>0.451443</td>
      <td>0.289447</td>
      <td>0.074576</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.037398</td>
      <td>0.050000</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Only show the marker genes information, without the heatmap
leiden_key = f'leiden_{optimal_resolution}'

# Check if we have marker genes information and display them
if f"rank_genes_{optimal_resolution}" in adata.uns:
    # Get top markers for each cluster (adjust n_genes as needed)
    n_top_genes = 20
    
    # Load and display top markers for each cluster in the optimal clustering
    markers_file = os.path.join(OUTPUT_DIR, 'my_cluster_analysis', 'marker_analysis', f'cluster_markers_res{optimal_resolution}.csv')
    if os.path.exists(markers_file):
        markers_df = pd.read_csv(markers_file)
        
        # Create a more readable format for marker genes by cluster
        top_markers_by_cluster = {}
        for cluster in sorted(markers_df['cluster'].unique()):
            cluster_markers = markers_df[markers_df['cluster'] == cluster].sort_values('pvals_adj').head(10)
            top_markers_by_cluster[cluster] = list(zip(
                cluster_markers['names'], 
                cluster_markers['logfoldchanges'].round(2),
                cluster_markers['pvals_adj'].apply(lambda x: f"{x:.2e}")
            ))
        
        # Display top markers for each cluster
        print(f"Top marker genes for each cluster at resolution {optimal_resolution}:")
        for cluster, markers in top_markers_by_cluster.items():
            print(f"\nCluster {cluster}:")
            for i, (gene, lfc, pval) in enumerate(markers, 1):
                print(f"  {i}. {gene} (log2FC: {lfc}, adj.p-val: {pval})")
else:
    print("No marker gene information available for the optimal clustering.")
```

    Top marker genes for each cluster at resolution 0.47:
    
    Cluster 0:
      1. Pfdn5 (log2FC: 3.02, adj.p-val: 8.64e-26)
      2. Bex2 (log2FC: 2.09, adj.p-val: 6.43e-22)
      3. Ndufa6 (log2FC: 3.6, adj.p-val: 5.25e-20)
      4. Plscr4 (log2FC: 2.19, adj.p-val: 9.77e-16)
      5. Arl10 (log2FC: -0.46, adj.p-val: 2.97e-15)
      6. Mesd (log2FC: 2.24, adj.p-val: 3.52e-15)
      7. Primpol (log2FC: 0.07, adj.p-val: 1.07e-13)
      8. Cops9 (log2FC: 1.36, adj.p-val: 5.43e-13)
      9. Snu13 (log2FC: 2.47, adj.p-val: 4.16e-12)
      10. Selenom (log2FC: 1.09, adj.p-val: 1.97e-11)
    
    Cluster 1:
      1. Rpl10 (log2FC: -0.41, adj.p-val: 1.00e+00)
      2. Srp19 (log2FC: 1.84, adj.p-val: 1.00e+00)
      3. Thap12 (log2FC: 1.27, adj.p-val: 1.00e+00)
      4. Ino80b (log2FC: -4.37, adj.p-val: 1.00e+00)
      5. Kcnj2 (log2FC: 2.34, adj.p-val: 1.00e+00)
      6. Prpf19 (log2FC: 4.0, adj.p-val: 1.00e+00)
      7. 2300009A05Rik (log2FC: 4.07, adj.p-val: 1.00e+00)
      8. Pgls (log2FC: -2.15, adj.p-val: 1.00e+00)
      9. Cope (log2FC: 2.08, adj.p-val: 1.00e+00)
      10. Kin (log2FC: 3.89, adj.p-val: 1.00e+00)
    
    Cluster 2:
      1. Ttc23l (log2FC: -2.94, adj.p-val: 1.00e+00)
      2. Rhox4b (log2FC: -5.18, adj.p-val: 1.00e+00)
      3. Soat2 (log2FC: -5.02, adj.p-val: 1.00e+00)
      4. Gm41002 (log2FC: -4.29, adj.p-val: 1.00e+00)
      5. Frmpd1os (log2FC: -3.4, adj.p-val: 1.00e+00)
      6. Gm17308 (log2FC: -9.21, adj.p-val: 1.00e+00)
      7. Gm26646 (log2FC: -5.31, adj.p-val: 1.00e+00)
      8. Gm11184 (log2FC: -3.97, adj.p-val: 1.00e+00)
      9. Dbhos (log2FC: -5.54, adj.p-val: 1.00e+00)
      10. Gm20513 (log2FC: -3.35, adj.p-val: 1.00e+00)
    
    Cluster 3:
      1. Gm10101 (log2FC: -3.84, adj.p-val: 9.96e-01)
      2. 1700003C15Rik (log2FC: -4.29, adj.p-val: 9.96e-01)
      3. Gm11770 (log2FC: -6.17, adj.p-val: 9.96e-01)
      4. Gm26576 (log2FC: -4.85, adj.p-val: 9.96e-01)
      5. Gm57030 (log2FC: -5.28, adj.p-val: 9.96e-01)
      6. Gm21860 (log2FC: -5.11, adj.p-val: 9.96e-01)
      7. Gm15326 (log2FC: -3.88, adj.p-val: 9.96e-01)
      8. Wfdc1 (log2FC: -3.7, adj.p-val: 9.96e-01)
      9. 4930534H18Rik (log2FC: -3.53, adj.p-val: 9.96e-01)
      10. Platr27 (log2FC: -2.28, adj.p-val: 9.96e-01)
    
    Cluster 4:
      1. Gm10101 (log2FC: -3.96, adj.p-val: 9.94e-01)
      2. Gm42205 (log2FC: -3.33, adj.p-val: 9.94e-01)
      3. Vmn2r98 (log2FC: -10.82, adj.p-val: 9.94e-01)
      4. Ccdc27 (log2FC: -3.71, adj.p-val: 9.94e-01)
      5. S100a9 (log2FC: -7.37, adj.p-val: 9.94e-01)
      6. Gm44773 (log2FC: -2.93, adj.p-val: 9.94e-01)
      7. Gm46416 (log2FC: -4.06, adj.p-val: 9.94e-01)
      8. Gm14409 (log2FC: -4.48, adj.p-val: 9.94e-01)
      9. Gm28180 (log2FC: -3.23, adj.p-val: 9.94e-01)
      10. Tcp10a (log2FC: -2.84, adj.p-val: 9.94e-01)
    
    Cluster 5:
      1. Col4a6 (log2FC: -0.68, adj.p-val: 1.00e+00)
      2. Abcb1a (log2FC: -1.22, adj.p-val: 1.00e+00)
      3. Gm26747 (log2FC: -4.39, adj.p-val: 1.00e+00)
      4. Or13a24 (log2FC: -1.8, adj.p-val: 1.00e+00)
      5. Gm45615 (log2FC: -8.93, adj.p-val: 1.00e+00)
      6. Gm45721 (log2FC: -3.87, adj.p-val: 1.00e+00)
      7. Spata25 (log2FC: -4.73, adj.p-val: 1.00e+00)
      8. Gm19935 (log2FC: -4.48, adj.p-val: 1.00e+00)
      9. Gm15562 (log2FC: -3.7, adj.p-val: 1.00e+00)
      10. Cox6b2 (log2FC: -3.1, adj.p-val: 1.00e+00)
    
    Cluster 6:
      1. Gm13595 (log2FC: -7.54, adj.p-val: 1.00e+00)
      2. 3300002P09Rik (log2FC: -5.87, adj.p-val: 1.00e+00)
      3. Gm28586 (log2FC: -4.39, adj.p-val: 1.00e+00)
      4. Gm57239 (log2FC: -5.3, adj.p-val: 1.00e+00)
      5. Lnp1 (log2FC: -5.87, adj.p-val: 1.00e+00)
      6. Gm20416 (log2FC: -7.25, adj.p-val: 1.00e+00)
      7. Gm10390 (log2FC: -2.24, adj.p-val: 1.00e+00)
      8. Gm41561 (log2FC: -7.62, adj.p-val: 1.00e+00)
      9. Gm17059 (log2FC: -3.8, adj.p-val: 1.00e+00)
      10. Or5w14 (log2FC: -4.6, adj.p-val: 1.00e+00)
    
    Cluster 7:
      1. Hspb11 (log2FC: 2.13, adj.p-val: 2.22e-01)
      2. Tnfaip1 (log2FC: 3.1, adj.p-val: 3.08e-01)
      3. H2az1 (log2FC: 6.75, adj.p-val: 4.10e-01)
      4. B230354K17Rik (log2FC: 2.73, adj.p-val: 5.15e-01)
      5. Acox3 (log2FC: 4.22, adj.p-val: 5.49e-01)
      6. Donson (log2FC: 6.02, adj.p-val: 5.60e-01)
      7. Zfp120 (log2FC: -0.89, adj.p-val: 5.74e-01)
      8. Nop14 (log2FC: 6.45, adj.p-val: 5.79e-01)
      9. Drap1 (log2FC: 4.98, adj.p-val: 5.89e-01)
      10. Nphp3 (log2FC: 4.1, adj.p-val: 5.92e-01)
    
    Cluster 8:
      1. Gm57275 (log2FC: -5.86, adj.p-val: 1.00e+00)
      2. Fndc3c1 (log2FC: -8.97, adj.p-val: 1.00e+00)
      3. Art2a (log2FC: -5.84, adj.p-val: 1.00e+00)
      4. Or14j7 (log2FC: -5.49, adj.p-val: 1.00e+00)
      5. Gm56856 (log2FC: -3.94, adj.p-val: 1.00e+00)
      6. 4930540M05Rik (log2FC: -4.31, adj.p-val: 1.00e+00)
      7. Gm14717 (log2FC: -6.7, adj.p-val: 1.00e+00)
      8. Gm30288 (log2FC: -9.0, adj.p-val: 1.00e+00)
      9. Gm46658 (log2FC: -5.71, adj.p-val: 1.00e+00)
      10. 1700021J08Rik (log2FC: -3.27, adj.p-val: 1.00e+00)
    
    Cluster 9:
      1. Stbd1 (log2FC: -5.64, adj.p-val: 1.00e+00)
      2. Or14j10 (log2FC: -5.6, adj.p-val: 1.00e+00)
      3. Gm11131 (log2FC: -5.11, adj.p-val: 1.00e+00)
      4. Pilra (log2FC: -3.44, adj.p-val: 1.00e+00)
      5. 9130409I23Rik (log2FC: -4.96, adj.p-val: 1.00e+00)
      6. Gm28988 (log2FC: -5.21, adj.p-val: 1.00e+00)
      7. Tbx22 (log2FC: -4.61, adj.p-val: 1.00e+00)
      8. Trgc4 (log2FC: -3.58, adj.p-val: 1.00e+00)
      9. Gm57156 (log2FC: -5.39, adj.p-val: 1.00e+00)
      10. C2cd4d (log2FC: -6.46, adj.p-val: 1.00e+00)
    
    Cluster 10:
      1. Gm14209 (log2FC: -5.18, adj.p-val: 9.97e-01)
      2. 1700003D09Rik (log2FC: -6.36, adj.p-val: 9.97e-01)
      3. Msx1 (log2FC: -5.55, adj.p-val: 9.97e-01)
      4. Ankrd53 (log2FC: -6.44, adj.p-val: 9.97e-01)
      5. Gm35909 (log2FC: -3.94, adj.p-val: 9.97e-01)
      6. Or2c1 (log2FC: -5.92, adj.p-val: 9.97e-01)
      7. Gm12462 (log2FC: -5.82, adj.p-val: 9.97e-01)
      8. Slc6a4 (log2FC: -4.16, adj.p-val: 9.97e-01)
      9. Gm56685 (log2FC: -6.0, adj.p-val: 9.97e-01)
      10. Apobr (log2FC: -5.6, adj.p-val: 9.97e-01)
    
    Cluster 11:
      1. Evi2a (log2FC: -2.61, adj.p-val: 1.00e+00)
      2. mt-Nd5 (log2FC: 8.2, adj.p-val: 1.00e+00)
      3. Bc1 (log2FC: 8.84, adj.p-val: 1.00e+00)
      4. Gm48742 (log2FC: 2.77, adj.p-val: 1.00e+00)
      5. Gm38973 (log2FC: 3.43, adj.p-val: 1.00e+00)
      6. Aamp (log2FC: 3.93, adj.p-val: 1.00e+00)
      7. Tns1 (log2FC: 3.11, adj.p-val: 1.00e+00)
      8. Crlf2 (log2FC: -0.73, adj.p-val: 1.00e+00)
      9. Smc4 (log2FC: 1.23, adj.p-val: 1.00e+00)
      10. Setdb2 (log2FC: -1.06, adj.p-val: 1.00e+00)
    
    Cluster 12:
      1. Gm40117 (log2FC: -3.22, adj.p-val: 1.00e+00)
      2. Slc6a13 (log2FC: -1.78, adj.p-val: 1.00e+00)
      3. Acox2 (log2FC: -2.51, adj.p-val: 1.00e+00)
      4. Rs1 (log2FC: -2.14, adj.p-val: 1.00e+00)
      5. Gm31223 (log2FC: -2.86, adj.p-val: 1.00e+00)
      6. 4932414N04Rik (log2FC: -2.26, adj.p-val: 1.00e+00)
      7. Hp (log2FC: -3.37, adj.p-val: 1.00e+00)
      8. Tagln2 (log2FC: -5.23, adj.p-val: 1.00e+00)
      9. Gm5617 (log2FC: -1.9, adj.p-val: 1.00e+00)
      10. Gm48250 (log2FC: -2.22, adj.p-val: 1.00e+00)
    
    Cluster 13:
      1. Pfdn5 (log2FC: 5.22, adj.p-val: 6.65e-01)
      2. Acot13 (log2FC: 1.74, adj.p-val: 1.00e+00)
      3. Arhgap29 (log2FC: 5.21, adj.p-val: 1.00e+00)
      4. Xpc (log2FC: 7.67, adj.p-val: 1.00e+00)
      5. Osgin2 (log2FC: 3.06, adj.p-val: 1.00e+00)
      6. Plxnb2 (log2FC: 0.09, adj.p-val: 1.00e+00)
      7. Ebag9 (log2FC: 6.13, adj.p-val: 1.00e+00)
      8. Gatad1 (log2FC: 4.0, adj.p-val: 1.00e+00)
      9. Ftsj1 (log2FC: 1.33, adj.p-val: 1.00e+00)
      10. Ttl (log2FC: 2.97, adj.p-val: 1.00e+00)
    
    Cluster 14:
      1. Vsig2 (log2FC: -4.23, adj.p-val: 9.99e-01)
      2. Ccdc105 (log2FC: -3.28, adj.p-val: 9.99e-01)
      3. Gm49944 (log2FC: -3.83, adj.p-val: 9.99e-01)
      4. Gm34418 (log2FC: -3.79, adj.p-val: 9.99e-01)
      5. Gm12446 (log2FC: -3.58, adj.p-val: 9.99e-01)
      6. H4c3 (log2FC: -3.14, adj.p-val: 9.99e-01)
      7. Esm1 (log2FC: -3.86, adj.p-val: 9.99e-01)
      8. Rassf10 (log2FC: -3.92, adj.p-val: 9.99e-01)
      9. Gm27198 (log2FC: -4.06, adj.p-val: 9.99e-01)
      10. Marveld3 (log2FC: -3.79, adj.p-val: 9.99e-01)
    
    Cluster 15:
      1. Txnip (log2FC: -4.21, adj.p-val: 9.98e-01)
      2. Gm56895 (log2FC: 0.35, adj.p-val: 9.98e-01)
      3. Gm40292 (log2FC: 0.43, adj.p-val: 9.98e-01)
      4. Or10g1b (log2FC: 0.23, adj.p-val: 9.99e-01)
      5. Gm57218 (log2FC: 0.27, adj.p-val: 9.99e-01)
      6. 1700006A11Rik (log2FC: 0.29, adj.p-val: 9.99e-01)
      7. Gm14205 (log2FC: 0.3, adj.p-val: 9.99e-01)
      8. Sirpb1a (log2FC: 0.26, adj.p-val: 9.99e-01)
      9. Ces1b (log2FC: 0.23, adj.p-val: 9.99e-01)
      10. S100a6 (log2FC: 0.15, adj.p-val: 9.99e-01)
    
    Cluster 16:
      1. Btbd17 (log2FC: -6.96, adj.p-val: 1.00e+00)
      2. Ddx60 (log2FC: -1.98, adj.p-val: 1.00e+00)
      3. Plxnb1 (log2FC: -0.1, adj.p-val: 1.00e+00)
      4. Rex1bd (log2FC: -2.28, adj.p-val: 1.00e+00)
      5. Slc35d2 (log2FC: -1.75, adj.p-val: 1.00e+00)
      6. Cacna2d4 (log2FC: -2.06, adj.p-val: 1.00e+00)
      7. Tlr1 (log2FC: -3.48, adj.p-val: 1.00e+00)
      8. S100a13 (log2FC: -0.95, adj.p-val: 1.00e+00)
      9. Gm31036 (log2FC: -1.09, adj.p-val: 1.00e+00)
      10. P2ry10b (log2FC: 0.15, adj.p-val: 1.00e+00)
    
    Cluster 17:
      1. P2ry6 (log2FC: 1.55, adj.p-val: 1.00e+00)
      2. Ddx60 (log2FC: -0.43, adj.p-val: 1.00e+00)
      3. Mpeg1 (log2FC: 2.65, adj.p-val: 1.00e+00)
      4. Alox5 (log2FC: -1.92, adj.p-val: 1.00e+00)
      5. Ppp1r3b (log2FC: -0.9, adj.p-val: 1.00e+00)
      6. S100a3 (log2FC: -0.41, adj.p-val: 1.00e+00)
      7. Tyrobp (log2FC: 2.61, adj.p-val: 1.00e+00)
      8. Il23a (log2FC: -2.49, adj.p-val: 1.00e+00)
      9. C1qtnf6 (log2FC: -2.32, adj.p-val: 1.00e+00)
      10. D030051J21Rik (log2FC: -1.35, adj.p-val: 1.00e+00)
    
    Cluster 18:
      1. Naip2 (log2FC: 1.08, adj.p-val: 1.00e+00)
      2. Vps9d1 (log2FC: 9.21, adj.p-val: 1.00e+00)
      3. Rin3 (log2FC: 2.58, adj.p-val: 1.00e+00)
      4. Gm39816 (log2FC: 2.17, adj.p-val: 1.00e+00)
      5. Slc39a3 (log2FC: -1.69, adj.p-val: 1.00e+00)
      6. Bcas2 (log2FC: 2.79, adj.p-val: 1.00e+00)
      7. Lig1 (log2FC: 0.63, adj.p-val: 1.00e+00)
      8. 2300009A05Rik (log2FC: 4.08, adj.p-val: 1.00e+00)
      9. 2810001A02Rik (log2FC: -0.66, adj.p-val: 1.00e+00)
      10. Srp14 (log2FC: 1.83, adj.p-val: 1.00e+00)
    
    Cluster 19:
      1. Crybb1 (log2FC: 0.33, adj.p-val: 9.98e-01)
      2. Il4ra (log2FC: -2.11, adj.p-val: 9.98e-01)
      3. Mmp14 (log2FC: -0.74, adj.p-val: 9.98e-01)
      4. Susd3 (log2FC: 1.55, adj.p-val: 9.98e-01)
      5. 1700006A11Rik (log2FC: 0.29, adj.p-val: 9.99e-01)
      6. Ces1b (log2FC: 0.23, adj.p-val: 9.99e-01)
      7. Sirpb1a (log2FC: 0.26, adj.p-val: 9.99e-01)
      8. Or10g1b (log2FC: 0.23, adj.p-val: 9.99e-01)
      9. Gm14205 (log2FC: 0.29, adj.p-val: 9.99e-01)
      10. Gm57218 (log2FC: 0.27, adj.p-val: 9.99e-01)
    
    Cluster 20:
      1. Gm39154 (log2FC: 0.32, adj.p-val: 9.98e-01)
      2. 4930526F13Rik (log2FC: 0.35, adj.p-val: 9.98e-01)
      3. 4930550C17Rik (log2FC: 0.35, adj.p-val: 9.98e-01)
      4. Gm34597 (log2FC: 0.32, adj.p-val: 9.98e-01)
      5. Ces1b (log2FC: 0.23, adj.p-val: 9.99e-01)
      6. Sirpb1a (log2FC: 0.26, adj.p-val: 9.99e-01)
      7. Gm14205 (log2FC: 0.29, adj.p-val: 9.99e-01)
      8. 1700006A11Rik (log2FC: 0.29, adj.p-val: 9.99e-01)
      9. Or10g1b (log2FC: 0.23, adj.p-val: 9.99e-01)
      10. Gm57218 (log2FC: 0.27, adj.p-val: 9.99e-01)
    
    Cluster 21:
      1. Stab1 (log2FC: 4.28, adj.p-val: 9.99e-01)
      2. Tec (log2FC: 5.06, adj.p-val: 9.99e-01)
      3. Oplah (log2FC: -0.81, adj.p-val: 9.99e-01)
      4. Sirpb1a (log2FC: 0.26, adj.p-val: 1.00e+00)
      5. Gm57218 (log2FC: 0.27, adj.p-val: 1.00e+00)
      6. Gm14205 (log2FC: 0.29, adj.p-val: 1.00e+00)
      7. Or10g1b (log2FC: 0.23, adj.p-val: 1.00e+00)
      8. Ces1b (log2FC: 0.23, adj.p-val: 1.00e+00)
      9. 1700006A11Rik (log2FC: 0.29, adj.p-val: 1.00e+00)
      10. Pcsk6 (log2FC: 8.78, adj.p-val: 1.00e+00)
    
    Cluster 22:
      1. Trim56 (log2FC: -1.69, adj.p-val: 9.99e-01)
      2. Tirap (log2FC: -1.89, adj.p-val: 9.99e-01)
      3. Rexo5 (log2FC: -1.09, adj.p-val: 9.99e-01)
      4. Mrpl36 (log2FC: -1.13, adj.p-val: 9.99e-01)
      5. Rnase4 (log2FC: 1.93, adj.p-val: 9.99e-01)
      6. Dnajc28 (log2FC: -4.95, adj.p-val: 9.99e-01)
      7. 2810459M11Rik (log2FC: -0.52, adj.p-val: 9.99e-01)
      8. Spocd1 (log2FC: -0.74, adj.p-val: 9.99e-01)
      9. Npepl1 (log2FC: 1.53, adj.p-val: 9.99e-01)
      10. 9130015G15Rik (log2FC: 1.67, adj.p-val: 9.99e-01)



```python
print("NOTE: Heatmap generation has been moved to a separate script.")
print("Please use the generate_marker_heatmaps.py script to create heatmaps from the saved .h5ad files.")

# %% [markdown]
# # 9. Summary and Conclusion
```

    NOTE: Heatmap generation has been moved to a separate script.
    Please use the generate_marker_heatmaps.py script to create heatmaps from the saved .h5ad files.



```python
# Load and display the analysis summary
summary_path = os.path.join(OUTPUT_DIR, 'my_cluster_analysis', 'analysis_summary.txt')
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        summary_text = f.read()
    
    from IPython.display import Markdown
    display(Markdown(f"```\n{summary_text}\n```"))
```


```
==================================================
CLUSTERING ANALYSIS SUMMARY
==================================================

Date: 2025-03-28 10:06:44
Analysis duration: 708.1 seconds

Resolutions tested: 10
Resolution range: 0.05 to 0.8

OPTIMAL CLUSTERING RESULT:
- Optimal resolution: 0.47
- Number of clusters: 23
- Silhouette score: 0.0933
- Davies-Bouldin score: 1.8212 (lower is better)
- Calinski-Harabasz score: 453.5
- Avg. marker genes per cluster: 10.0
- Marker gene score: 1.0000
- Overall quality score: 0.7624

Results saved to:
- /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Nestin_Mut_adult_3/my_cluster_analysis

All clustering resolutions saved in the AnnData object:
- leiden_0.05: 7 clusters
- leiden_0.13: 12 clusters
- leiden_0.22: 14 clusters
- leiden_0.3: 18 clusters
- leiden_0.38: 19 clusters
- leiden_0.47: 23 clusters
- leiden_0.55: 25 clusters
- leiden_0.63: 25 clusters
- leiden_0.72: 26 clusters
- leiden_0.8: 30 clusters

```



```python
# Print final summary
print(f"\n{'='*50}")
print(f"CLUSTERING ANALYSIS COMPLETED")
print(f"{'='*50}")
print(f"Sample: {SAMPLE_NAME}")
print(f"Optimal resolution: {optimal_resolution}")
print(f"Number of clusters: {len(adata.obs[f'leiden_{optimal_resolution}'].unique())}")
print(f"Total cells analyzed: {adata.n_obs}")
print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")
print(f"{'='*50}")



```

    
    ==================================================
    CLUSTERING ANALYSIS COMPLETED
    ==================================================
    Sample: Nestin_Mut
    Optimal resolution: 0.47
    Number of clusters: 23
    Total cells analyzed: 9237
    Results saved to: /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Nestin_Mut_adult_3
    ==================================================

