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
SAMPLE_NAME = "Nestin_Ctrl"  # This will be replaced with the actual sample name
# SAMPLE_NAME = "Emx1_Ctrl"
print(f"Processing sample: {SAMPLE_NAME}")

# %% [markdown]
# # 1. Setup and Data Loading
```

    Processing sample: Nestin_Ctrl



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

    ... reading from cache file cache/beegfs-scratch-ric.broccoli-kubacki.michal-SRF_Linda_RNA-cellranger_final_count_data-cellranger_counts_R26_Nestin_Ctrl_adult_2-outs-filtered_feature_bc_matrix-matrix.h5ad


    Shape of loaded data: (9555, 33696)



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

    filtered out 6132 genes that are detected in less than 3 cells



    
![png](Nestin_Ctrl_files/Nestin_Ctrl_5_1.png)
    



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

    Number of cells after filtering: 9512
    Number of genes after filtering: 27564



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


    Number of highly variable genes: 6251



    <Figure size 1000x800 with 0 Axes>



    
![png](Nestin_Ctrl_files/Nestin_Ctrl_7_7.png)
    



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


        finished (0:00:20)



    
![png](Nestin_Ctrl_files/Nestin_Ctrl_10_4.png)
    



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



    
![png](Nestin_Ctrl_files/Nestin_Ctrl_11_6.png)
    



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

    X matrix values (first cell): [-0.87605834 -0.2857414  -0.6030637  -0.02675853 -0.08374193]
    Should be log1p transformed values (~0-5 range)
    Raw values: [1. 0. 0. 0. 0.]
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


        finished: found 9 clusters and added
        'leiden_0.05', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  10%|█         | 1/10 [00:00<00:03,  2.38it/s]

    running Leiden clustering


        finished: found 10 clusters and added
        'leiden_0.13', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  20%|██        | 2/10 [00:00<00:03,  2.60it/s]

    running Leiden clustering


        finished: found 14 clusters and added
        'leiden_0.22', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  30%|███       | 3/10 [00:01<00:02,  2.43it/s]

    running Leiden clustering


        finished: found 19 clusters and added
        'leiden_0.3', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  40%|████      | 4/10 [00:01<00:02,  2.25it/s]

    running Leiden clustering


        finished: found 20 clusters and added
        'leiden_0.38', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  50%|█████     | 5/10 [00:02<00:02,  2.40it/s]

    running Leiden clustering


        finished: found 22 clusters and added
        'leiden_0.47', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  60%|██████    | 6/10 [00:02<00:01,  2.41it/s]

    running Leiden clustering


        finished: found 22 clusters and added
        'leiden_0.55', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  70%|███████   | 7/10 [00:02<00:01,  2.43it/s]

    running Leiden clustering


        finished: found 23 clusters and added
        'leiden_0.63', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  80%|████████  | 8/10 [00:03<00:00,  2.28it/s]

    running Leiden clustering


        finished: found 24 clusters and added
        'leiden_0.72', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  90%|█████████ | 9/10 [00:04<00:00,  2.01it/s]

    running Leiden clustering


        finished: found 28 clusters and added
        'leiden_0.8', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings: 100%|██████████| 10/10 [00:04<00:00,  1.91it/s]

    Computing clusterings: 100%|██████████| 10/10 [00:04<00:00,  2.17it/s]

    


    
    Step 2: Identifying marker genes for each clustering resolution...


    Processing resolutions:   0%|          | 0/10 [00:00<?, ?it/s]

    
    Analyzing resolution 0.05:
    ranking genes


        finished: added to `.uns['rank_genes_0.05']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:41)


      ✓ Identified differentially expressed genes


    Processing resolutions:  10%|█         | 1/10 [00:44<06:38, 44.29s/it]

      ✓ Generated marker ranking plot
      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.05.csv
    
    Analyzing resolution 0.13:
    ranking genes


        finished: added to `.uns['rank_genes_0.13']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:45)


      ✓ Identified differentially expressed genes


    Processing resolutions:  20%|██        | 2/10 [01:31<06:10, 46.25s/it]

      ✓ Generated marker ranking plot
      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.13.csv
    
    Analyzing resolution 0.22:
    ranking genes


        finished: added to `.uns['rank_genes_0.22']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:57)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  30%|███       | 3/10 [02:33<06:11, 53.14s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.22.csv
    
    Analyzing resolution 0.3:
    ranking genes


        finished: added to `.uns['rank_genes_0.3']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:13)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  40%|████      | 4/10 [03:51<06:17, 62.93s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.3.csv
    
    Analyzing resolution 0.38:
    ranking genes


        finished: added to `.uns['rank_genes_0.38']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:16)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  50%|█████     | 5/10 [05:12<05:48, 69.71s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.38.csv
    
    Analyzing resolution 0.47:
    ranking genes


        finished: added to `.uns['rank_genes_0.47']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:23)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  60%|██████    | 6/10 [06:41<05:03, 75.96s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.47.csv
    
    Analyzing resolution 0.55:
    ranking genes


        finished: added to `.uns['rank_genes_0.55']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:23)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  70%|███████   | 7/10 [08:09<04:00, 80.04s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.55.csv
    
    Analyzing resolution 0.63:
    ranking genes


        finished: added to `.uns['rank_genes_0.63']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:27)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  80%|████████  | 8/10 [09:42<02:48, 84.26s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.63.csv
    
    Analyzing resolution 0.72:
    ranking genes


        finished: added to `.uns['rank_genes_0.72']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:30)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  90%|█████████ | 9/10 [11:19<01:28, 88.02s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.72.csv
    
    Analyzing resolution 0.8:
    ranking genes


        finished: added to `.uns['rank_genes_0.8']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:01:41)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions: 100%|██████████| 10/10 [13:07<00:00, 94.24s/it]

    Processing resolutions: 100%|██████████| 10/10 [13:07<00:00, 78.72s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.8.csv
    
    Summary comparison saved to my_cluster_analysis/marker_analysis/resolution_comparison_summary.csv


    


    
    Analysis complete. Results saved to my_cluster_analysis/marker_analysis/
    
    Step 3: Evaluating clustering quality and selecting optimal resolution...
    Evaluating clustering metrics across resolutions...


    Evaluating resolutions:   0%|          | 0/10 [00:00<?, ?it/s]

    Evaluating resolutions:  10%|█         | 1/10 [00:00<00:06,  1.42it/s]

    Evaluating resolutions:  20%|██        | 2/10 [00:01<00:05,  1.43it/s]

    Evaluating resolutions:  30%|███       | 3/10 [00:02<00:04,  1.45it/s]

    Evaluating resolutions:  40%|████      | 4/10 [00:02<00:04,  1.46it/s]

    Evaluating resolutions:  50%|█████     | 5/10 [00:03<00:03,  1.47it/s]

    Evaluating resolutions:  60%|██████    | 6/10 [00:04<00:02,  1.47it/s]

    Evaluating resolutions:  70%|███████   | 7/10 [00:04<00:02,  1.48it/s]

    Evaluating resolutions:  80%|████████  | 8/10 [00:05<00:01,  1.48it/s]

    Evaluating resolutions:  90%|█████████ | 9/10 [00:06<00:00,  1.48it/s]

    Evaluating resolutions: 100%|██████████| 10/10 [00:06<00:00,  1.48it/s]

    Evaluating resolutions: 100%|██████████| 10/10 [00:06<00:00,  1.47it/s]

    


    
    Optimal clustering resolution: 0.05
    Optimal number of clusters: 9
    Metrics saved to my_cluster_analysis/evaluation/clustering_quality_metrics.csv


    Detailed metric analysis saved to my_cluster_analysis/evaluation/metric_details
    
    Analysis complete in 804.3 seconds!
    Optimal resolution: 0.05 (9 clusters)
    All clustering resolutions have been preserved in the AnnData object
    Full results saved to /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Nestin_Ctrl_adult_2/my_cluster_analysis



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
    1. Resolution: 0.05, Overall Score: 0.7835592805000358
    2. Resolution: 0.13, Overall Score: 0.6847978352724843
    3. Resolution: 0.47, Overall Score: 0.6389683116341831
    4. Resolution: 0.55, Overall Score: 0.6162478295304149
    5. Resolution: 0.63, Overall Score: 0.5742047112450047
    6. Resolution: 0.38, Overall Score: 0.5422770170885554
    7. Resolution: 0.8, Overall Score: 0.4423699925343196
    8. Resolution: 0.72, Overall Score: 0.4328638359030323
    9. Resolution: 0.3, Overall Score: 0.4162629336098324
    10. Resolution: 0.22, Overall Score: 0.4010395919418766



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

    [0.05, 0.13, 0.47]



    
![png](Nestin_Ctrl_files/Nestin_Ctrl_16_1.png)
    



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
      - leiden_0.05: 9 clusters
      - leiden_0.13: 10 clusters
      - leiden_0.22: 14 clusters
      - leiden_0.3: 19 clusters
      - leiden_0.38: 20 clusters
      - leiden_0.47: 22 clusters
      - leiden_0.55: 22 clusters
      - leiden_0.63: 23 clusters
      - leiden_0.72: 24 clusters
      - leiden_0.8: 28 clusters
    
    Saving processed AnnData object to: /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Nestin_Ctrl_adult_2/Nestin_Ctrl_processed.h5ad


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



    
![png](Nestin_Ctrl_files/Nestin_Ctrl_18_1.png)
    



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
      <td>9</td>
      <td>0.146817</td>
      <td>-1.419791</td>
      <td>1.000000</td>
      <td>0.783559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.13</td>
      <td>10</td>
      <td>0.133016</td>
      <td>-1.655558</td>
      <td>1.000000</td>
      <td>0.684798</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.22</td>
      <td>14</td>
      <td>0.127792</td>
      <td>-1.755710</td>
      <td>0.500000</td>
      <td>0.401040</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.30</td>
      <td>19</td>
      <td>0.133694</td>
      <td>-1.784931</td>
      <td>0.666667</td>
      <td>0.416263</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.38</td>
      <td>20</td>
      <td>0.132366</td>
      <td>-1.741746</td>
      <td>1.000000</td>
      <td>0.542277</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.47</td>
      <td>22</td>
      <td>0.144350</td>
      <td>-1.696835</td>
      <td>1.000000</td>
      <td>0.638968</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.55</td>
      <td>22</td>
      <td>0.144642</td>
      <td>-1.844534</td>
      <td>1.000000</td>
      <td>0.616248</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.63</td>
      <td>23</td>
      <td>0.141406</td>
      <td>-1.825000</td>
      <td>1.000000</td>
      <td>0.574205</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.72</td>
      <td>24</td>
      <td>0.135913</td>
      <td>-1.917753</td>
      <td>0.800000</td>
      <td>0.432864</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.80</td>
      <td>28</td>
      <td>0.137394</td>
      <td>-1.957485</td>
      <td>0.833333</td>
      <td>0.442370</td>
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



    
![png](Nestin_Ctrl_files/Nestin_Ctrl_20_1.png)
    



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



    
![png](Nestin_Ctrl_files/Nestin_Ctrl_21_1.png)
    


    Individual metrics across resolutions:



    
![png](Nestin_Ctrl_files/Nestin_Ctrl_21_3.png)
    



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
      <td>9</td>
      <td>0.783559</td>
      <td>0.350000</td>
      <td>0.220000</td>
      <td>0.220000</td>
      <td>0.0</td>
      <td>0.080000</td>
      <td>0.073424</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.13</td>
      <td>10</td>
      <td>0.684798</td>
      <td>0.096109</td>
      <td>0.123535</td>
      <td>0.195051</td>
      <td>0.0</td>
      <td>0.080000</td>
      <td>0.080000</td>
      <td>0.012199</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.22</td>
      <td>14</td>
      <td>0.401040</td>
      <td>0.000000</td>
      <td>0.082557</td>
      <td>0.110741</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.070912</td>
      <td>0.033974</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.30</td>
      <td>19</td>
      <td>0.416263</td>
      <td>0.108578</td>
      <td>0.070601</td>
      <td>0.047012</td>
      <td>0.0</td>
      <td>0.026667</td>
      <td>0.020484</td>
      <td>0.045431</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.38</td>
      <td>20</td>
      <td>0.542277</td>
      <td>0.084152</td>
      <td>0.088270</td>
      <td>0.047228</td>
      <td>0.0</td>
      <td>0.080000</td>
      <td>0.006044</td>
      <td>0.044878</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.47</td>
      <td>22</td>
      <td>0.638968</td>
      <td>0.304623</td>
      <td>0.106646</td>
      <td>0.034646</td>
      <td>0.0</td>
      <td>0.080000</td>
      <td>0.008419</td>
      <td>0.043286</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.55</td>
      <td>22</td>
      <td>0.616248</td>
      <td>0.309984</td>
      <td>0.046214</td>
      <td>0.036109</td>
      <td>0.0</td>
      <td>0.080000</td>
      <td>0.006769</td>
      <td>0.044764</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.63</td>
      <td>23</td>
      <td>0.574205</td>
      <td>0.250455</td>
      <td>0.054207</td>
      <td>0.026902</td>
      <td>0.0</td>
      <td>0.080000</td>
      <td>0.001708</td>
      <td>0.043932</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.72</td>
      <td>24</td>
      <td>0.432864</td>
      <td>0.149394</td>
      <td>0.016257</td>
      <td>0.019772</td>
      <td>0.0</td>
      <td>0.048000</td>
      <td>0.007198</td>
      <td>0.046117</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.80</td>
      <td>28</td>
      <td>0.442370</td>
      <td>0.176641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.053333</td>
      <td>0.000000</td>
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

    Top marker genes for each cluster at resolution 0.05:
    
    Cluster 0:
      1. Pla2g15 (log2FC: 3.44, adj.p-val: 7.96e-66)
      2. Tcof1 (log2FC: -2.08, adj.p-val: 6.20e-59)
      3. Flcn (log2FC: -1.85, adj.p-val: 3.96e-55)
      4. Lratd1 (log2FC: -0.12, adj.p-val: 3.27e-51)
      5. Eef1e1 (log2FC: -1.04, adj.p-val: 2.08e-48)
      6. Rpp21 (log2FC: -1.64, adj.p-val: 3.66e-47)
      7. Usf1 (log2FC: 0.86, adj.p-val: 2.35e-46)
      8. Dph3 (log2FC: -4.62, adj.p-val: 2.94e-45)
      9. Rnf181 (log2FC: -2.87, adj.p-val: 1.23e-42)
      10. Pex11b (log2FC: -1.66, adj.p-val: 2.04e-42)
    
    Cluster 1:
      1. H2-Q4 (log2FC: -2.68, adj.p-val: 9.98e-01)
      2. Myct1 (log2FC: -5.86, adj.p-val: 9.98e-01)
      3. Cdkn2b (log2FC: -2.28, adj.p-val: 9.98e-01)
      4. Gm14553 (log2FC: -2.3, adj.p-val: 9.98e-01)
      5. CJ186046Rik (log2FC: -3.54, adj.p-val: 9.98e-01)
      6. Rtp3 (log2FC: -3.08, adj.p-val: 9.98e-01)
      7. Ccrl2 (log2FC: -1.37, adj.p-val: 9.98e-01)
      8. 4933402J07Rik (log2FC: -4.17, adj.p-val: 9.98e-01)
      9. Gja4 (log2FC: -4.57, adj.p-val: 9.98e-01)
      10. Tfap2a (log2FC: -2.0, adj.p-val: 9.98e-01)
    
    Cluster 2:
      1. Irf2bp1 (log2FC: -5.9, adj.p-val: 6.69e-01)
      2. ENSMUSG00000121304 (log2FC: 0.76, adj.p-val: 1.00e+00)
      3. H1f0 (log2FC: 1.16, adj.p-val: 1.00e+00)
      4. H1f3 (log2FC: -0.14, adj.p-val: 1.00e+00)
      5. Sdf2l1 (log2FC: 2.22, adj.p-val: 1.00e+00)
      6. H2ac18 (log2FC: -1.27, adj.p-val: 1.00e+00)
      7. Lor (log2FC: -1.58, adj.p-val: 1.00e+00)
      8. Ebp (log2FC: 6.55, adj.p-val: 1.00e+00)
      9. Mrpl34 (log2FC: 2.53, adj.p-val: 1.00e+00)
      10. Apex1 (log2FC: 2.36, adj.p-val: 1.00e+00)
    
    Cluster 3:
      1. Gpha2 (log2FC: -6.47, adj.p-val: 9.99e-01)
      2. Gm46658 (log2FC: -4.67, adj.p-val: 9.99e-01)
      3. Cyp24a1 (log2FC: -6.65, adj.p-val: 9.99e-01)
      4. Gm12867 (log2FC: -7.54, adj.p-val: 9.99e-01)
      5. Gm42457 (log2FC: -3.21, adj.p-val: 9.99e-01)
      6. Gm10415 (log2FC: -4.57, adj.p-val: 9.99e-01)
      7. Cd79b (log2FC: -2.13, adj.p-val: 9.99e-01)
      8. Ces1f (log2FC: -3.25, adj.p-val: 9.99e-01)
      9. Banf2os (log2FC: -3.43, adj.p-val: 9.99e-01)
      10. Retnlg (log2FC: -2.15, adj.p-val: 9.99e-01)
    
    Cluster 4:
      1. H2-Ab1 (log2FC: -2.1, adj.p-val: 9.99e-01)
      2. Fam151a (log2FC: -6.09, adj.p-val: 9.99e-01)
      3. Ascl2 (log2FC: -1.07, adj.p-val: 9.99e-01)
      4. Gm56927 (log2FC: -7.61, adj.p-val: 9.99e-01)
      5. Epx (log2FC: -5.09, adj.p-val: 9.99e-01)
      6. Gm35551 (log2FC: -5.61, adj.p-val: 9.99e-01)
      7. Cxcl11 (log2FC: -5.9, adj.p-val: 9.99e-01)
      8. 4833403J16Rik (log2FC: -0.05, adj.p-val: 9.99e-01)
      9. Gm45218 (log2FC: -0.5, adj.p-val: 9.99e-01)
      10. Rbm46os (log2FC: -1.68, adj.p-val: 9.99e-01)
    
    Cluster 5:
      1. Gm49984 (log2FC: -5.83, adj.p-val: 9.98e-01)
      2. Gm32652 (log2FC: -4.71, adj.p-val: 9.98e-01)
      3. Asgr2 (log2FC: -7.48, adj.p-val: 9.98e-01)
      4. Sspnos (log2FC: -3.94, adj.p-val: 9.98e-01)
      5. Gm48709 (log2FC: -5.35, adj.p-val: 9.98e-01)
      6. Gm14453 (log2FC: -3.21, adj.p-val: 9.98e-01)
      7. 1810010K12Rik (log2FC: -3.51, adj.p-val: 9.98e-01)
      8. Gm56879 (log2FC: -9.07, adj.p-val: 9.98e-01)
      9. Gm57258 (log2FC: -4.6, adj.p-val: 9.98e-01)
      10. Pbp2 (log2FC: -5.03, adj.p-val: 9.98e-01)
    
    Cluster 6:
      1. Gm49808 (log2FC: -4.26, adj.p-val: 9.99e-01)
      2. F530104D19Rik (log2FC: -3.03, adj.p-val: 9.99e-01)
      3. Ankrd66 (log2FC: -1.93, adj.p-val: 9.99e-01)
      4. Atp6v0d2 (log2FC: -6.73, adj.p-val: 9.99e-01)
      5. 1700045H11Rik (log2FC: -2.92, adj.p-val: 9.99e-01)
      6. Gata2 (log2FC: -1.9, adj.p-val: 9.99e-01)
      7. Gm36486 (log2FC: -3.74, adj.p-val: 9.99e-01)
      8. Gm1110 (log2FC: -5.89, adj.p-val: 9.99e-01)
      9. Ly9 (log2FC: -2.71, adj.p-val: 9.99e-01)
      10. Gm38410 (log2FC: -2.59, adj.p-val: 9.99e-01)
    
    Cluster 7:
      1. Cd74 (log2FC: -1.82, adj.p-val: 9.99e-01)
      2. Gbp6 (log2FC: -0.66, adj.p-val: 9.99e-01)
      3. Tnfsf8 (log2FC: -0.41, adj.p-val: 9.99e-01)
      4. Nxph4 (log2FC: -0.47, adj.p-val: 9.99e-01)
      5. Dnajc5g (log2FC: -0.87, adj.p-val: 9.99e-01)
      6. 2810414N06Rik (log2FC: -5.21, adj.p-val: 9.99e-01)
      7. Gm10635 (log2FC: -0.09, adj.p-val: 9.99e-01)
      8. Gm57009 (log2FC: -0.39, adj.p-val: 9.99e-01)
      9. Raet1e (log2FC: -4.01, adj.p-val: 9.99e-01)
      10. Gm20560 (log2FC: 0.0, adj.p-val: 1.00e+00)
    
    Cluster 8:
      1. Cyyr1 (log2FC: 0.32, adj.p-val: 1.00e+00)
      2. F13a1 (log2FC: -1.07, adj.p-val: 1.00e+00)
      3. Top2a (log2FC: -0.39, adj.p-val: 1.00e+00)
      4. Slc14a1 (log2FC: 3.08, adj.p-val: 1.00e+00)
      5. ENSMUSG00000121501 (log2FC: -0.21, adj.p-val: 1.00e+00)
      6. Mlph (log2FC: 1.03, adj.p-val: 1.00e+00)
      7. H2ac24 (log2FC: -4.3, adj.p-val: 1.00e+00)
      8. Chrnd (log2FC: 0.16, adj.p-val: 1.00e+00)
      9. Myog (log2FC: 0.16, adj.p-val: 1.00e+00)
      10. Gm20560 (log2FC: 0.0, adj.p-val: 1.00e+00)



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

Date: 2025-03-28 10:08:23
Analysis duration: 804.3 seconds

Resolutions tested: 10
Resolution range: 0.05 to 0.8

OPTIMAL CLUSTERING RESULT:
- Optimal resolution: 0.05
- Number of clusters: 9
- Silhouette score: 0.1468
- Davies-Bouldin score: 1.4198 (lower is better)
- Calinski-Harabasz score: 1063.5
- Avg. marker genes per cluster: 10.0
- Marker gene score: 1.0000
- Overall quality score: 0.7836

Results saved to:
- /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Nestin_Ctrl_adult_2/my_cluster_analysis

All clustering resolutions saved in the AnnData object:
- leiden_0.05: 9 clusters
- leiden_0.13: 10 clusters
- leiden_0.22: 14 clusters
- leiden_0.3: 19 clusters
- leiden_0.38: 20 clusters
- leiden_0.47: 22 clusters
- leiden_0.55: 22 clusters
- leiden_0.63: 23 clusters
- leiden_0.72: 24 clusters
- leiden_0.8: 28 clusters

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
    Sample: Nestin_Ctrl
    Optimal resolution: 0.05
    Number of clusters: 9
    Total cells analyzed: 9512
    Results saved to: /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Nestin_Ctrl_adult_2
    ==================================================

