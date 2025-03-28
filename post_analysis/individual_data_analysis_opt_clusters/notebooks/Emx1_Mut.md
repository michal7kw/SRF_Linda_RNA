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
SAMPLE_NAME = "Emx1_Mut"  # This will be replaced with the actual sample name
# SAMPLE_NAME = "Emx1_Ctrl"
print(f"Processing sample: {SAMPLE_NAME}")

# %% [markdown]
# # 1. Setup and Data Loading
```

    Processing sample: Emx1_Mut



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

    ... reading from cache file cache/beegfs-scratch-ric.broccoli-kubacki.michal-SRF_Linda_RNA-cellranger_final_count_data-cellranger_counts_R26_Emx1_Mut_adult_1-outs-filtered_feature_bc_matrix-matrix.h5ad


    Shape of loaded data: (6567, 33696)



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

    filtered out 7394 genes that are detected in less than 3 cells



    
![png](Emx1_Mut_files/Emx1_Mut_5_1.png)
    



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

    Number of cells after filtering: 6242
    Number of genes after filtering: 26302



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


    Number of highly variable genes: 6712



    <Figure size 1000x800 with 0 Axes>



    
![png](Emx1_Mut_files/Emx1_Mut_7_7.png)
    



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



    
![png](Emx1_Mut_files/Emx1_Mut_10_4.png)
    



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
        `.obsp['connectivities']`, weighted adjacency matrix (0:00:06)


    computing UMAP


        finished: added
        'X_umap', UMAP coordinates (adata.obsm)
        'umap', UMAP parameters (adata.uns) (0:00:12)



    <Figure size 1000x800 with 0 Axes>



    
![png](Emx1_Mut_files/Emx1_Mut_11_6.png)
    



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

    X matrix values (first cell): [-0.10197631 -0.31850374  0.5213983  -0.03602417 -0.09663428]
    Should be log1p transformed values (~0-5 range)
    Raw values: [12.  0.  1.  0.  0.]
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


        finished: found 6 clusters and added
        'leiden_0.05', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  10%|█         | 1/10 [00:00<00:01,  4.91it/s]

    running Leiden clustering


        finished: found 12 clusters and added
        'leiden_0.13', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  20%|██        | 2/10 [00:00<00:01,  4.06it/s]

    running Leiden clustering


        finished: found 14 clusters and added
        'leiden_0.22', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  30%|███       | 3/10 [00:00<00:01,  4.22it/s]

    running Leiden clustering


        finished: found 19 clusters and added
        'leiden_0.3', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  40%|████      | 4/10 [00:00<00:01,  4.30it/s]

    running Leiden clustering


        finished: found 21 clusters and added
        'leiden_0.38', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  50%|█████     | 5/10 [00:01<00:01,  4.33it/s]

    running Leiden clustering


        finished: found 22 clusters and added
        'leiden_0.47', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  60%|██████    | 6/10 [00:01<00:01,  3.93it/s]

    running Leiden clustering


        finished: found 23 clusters and added
        'leiden_0.55', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  70%|███████   | 7/10 [00:01<00:00,  3.85it/s]

    running Leiden clustering


        finished: found 23 clusters and added
        'leiden_0.63', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  80%|████████  | 8/10 [00:01<00:00,  4.05it/s]

    running Leiden clustering


        finished: found 25 clusters and added
        'leiden_0.72', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings:  90%|█████████ | 9/10 [00:02<00:00,  3.93it/s]

    running Leiden clustering


        finished: found 26 clusters and added
        'leiden_0.8', the cluster labels (adata.obs, categorical) (0:00:00)


    Computing clusterings: 100%|██████████| 10/10 [00:02<00:00,  3.96it/s]

    Computing clusterings: 100%|██████████| 10/10 [00:02<00:00,  4.05it/s]

    


    
    Step 2: Identifying marker genes for each clustering resolution...


    Processing resolutions:   0%|          | 0/10 [00:00<?, ?it/s]

    
    Analyzing resolution 0.05:
    ranking genes


        finished: added to `.uns['rank_genes_0.05']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:18)


      ✓ Identified differentially expressed genes


    Processing resolutions:  10%|█         | 1/10 [00:19<02:58, 19.89s/it]

      ✓ Generated marker ranking plot
      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.05.csv
    
    Analyzing resolution 0.13:
    ranking genes


        finished: added to `.uns['rank_genes_0.13']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:29)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  20%|██        | 2/10 [00:51<03:36, 27.03s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.13.csv
    
    Analyzing resolution 0.22:
    ranking genes


        finished: added to `.uns['rank_genes_0.22']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:32)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  30%|███       | 3/10 [01:28<03:39, 31.35s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.22.csv
    
    Analyzing resolution 0.3:
    ranking genes


        finished: added to `.uns['rank_genes_0.3']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:41)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  40%|████      | 4/10 [02:14<03:44, 37.36s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.3.csv
    
    Analyzing resolution 0.38:
    ranking genes


        finished: added to `.uns['rank_genes_0.38']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:45)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  50%|█████     | 5/10 [03:05<03:31, 42.26s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.38.csv
    
    Analyzing resolution 0.47:
    ranking genes


        finished: added to `.uns['rank_genes_0.47']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:47)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  60%|██████    | 6/10 [03:58<03:03, 45.89s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.47.csv
    
    Analyzing resolution 0.55:
    ranking genes


        finished: added to `.uns['rank_genes_0.55']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:49)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  70%|███████   | 7/10 [04:53<02:26, 48.79s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.55.csv
    
    Analyzing resolution 0.63:
    ranking genes


        finished: added to `.uns['rank_genes_0.63']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:48)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  80%|████████  | 8/10 [05:47<01:41, 50.52s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.63.csv
    
    Analyzing resolution 0.72:
    ranking genes


        finished: added to `.uns['rank_genes_0.72']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:52)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions:  90%|█████████ | 9/10 [06:46<00:52, 52.94s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.72.csv
    
    Analyzing resolution 0.8:
    ranking genes


        finished: added to `.uns['rank_genes_0.8']`
        'names', sorted np.recarray to be indexed by group ids
        'scores', sorted np.recarray to be indexed by group ids
        'logfoldchanges', sorted np.recarray to be indexed by group ids
        'pvals', sorted np.recarray to be indexed by group ids
        'pvals_adj', sorted np.recarray to be indexed by group ids (0:00:54)


      ✓ Identified differentially expressed genes


      ✓ Generated marker ranking plot


    Processing resolutions: 100%|██████████| 10/10 [07:47<00:00, 55.50s/it]

    Processing resolutions: 100%|██████████| 10/10 [07:47<00:00, 46.73s/it]

      ✓ Saved 10 markers per cluster to my_cluster_analysis/marker_analysis/cluster_markers_res0.8.csv
    
    Summary comparison saved to my_cluster_analysis/marker_analysis/resolution_comparison_summary.csv


    


    
    Analysis complete. Results saved to my_cluster_analysis/marker_analysis/
    
    Step 3: Evaluating clustering quality and selecting optimal resolution...
    Evaluating clustering metrics across resolutions...


    Evaluating resolutions:   0%|          | 0/10 [00:00<?, ?it/s]

    Evaluating resolutions:  10%|█         | 1/10 [00:00<00:02,  3.05it/s]

    Evaluating resolutions:  20%|██        | 2/10 [00:00<00:02,  3.18it/s]

    Evaluating resolutions:  30%|███       | 3/10 [00:00<00:02,  3.22it/s]

    Evaluating resolutions:  40%|████      | 4/10 [00:01<00:01,  3.22it/s]

    Evaluating resolutions:  50%|█████     | 5/10 [00:01<00:01,  3.21it/s]

    Evaluating resolutions:  60%|██████    | 6/10 [00:01<00:01,  3.22it/s]

    Evaluating resolutions:  70%|███████   | 7/10 [00:02<00:00,  3.24it/s]

    Evaluating resolutions:  80%|████████  | 8/10 [00:02<00:00,  3.25it/s]

    Evaluating resolutions:  90%|█████████ | 9/10 [00:02<00:00,  3.26it/s]

    Evaluating resolutions: 100%|██████████| 10/10 [00:03<00:00,  3.27it/s]

    Evaluating resolutions: 100%|██████████| 10/10 [00:03<00:00,  3.24it/s]

    


    
    Optimal clustering resolution: 0.47
    Optimal number of clusters: 22
    Metrics saved to my_cluster_analysis/evaluation/clustering_quality_metrics.csv


    Detailed metric analysis saved to my_cluster_analysis/evaluation/metric_details
    
    Analysis complete in 478.4 seconds!
    Optimal resolution: 0.47 (22 clusters)
    All clustering resolutions have been preserved in the AnnData object
    Full results saved to /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Emx1_Mut_adult_1/my_cluster_analysis



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
    1. Resolution: 0.47, Overall Score: 0.8687881107799215
    2. Resolution: 0.38, Overall Score: 0.8655349205275662
    3. Resolution: 0.3, Overall Score: 0.832956277634536
    4. Resolution: 0.22, Overall Score: 0.7794034222738881
    5. Resolution: 0.13, Overall Score: 0.7290829567284057
    6. Resolution: 0.55, Overall Score: 0.684034511500088
    7. Resolution: 0.63, Overall Score: 0.6821194431452159
    8. Resolution: 0.72, Overall Score: 0.4095601029712317
    9. Resolution: 0.8, Overall Score: 0.3991204520695031
    10. Resolution: 0.05, Overall Score: 0.3728038039915886



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

    [0.47, 0.38, 0.3]



    
![png](Emx1_Mut_files/Emx1_Mut_16_1.png)
    



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
      - leiden_0.05: 6 clusters
      - leiden_0.13: 12 clusters
      - leiden_0.22: 14 clusters
      - leiden_0.3: 19 clusters
      - leiden_0.38: 21 clusters
      - leiden_0.47: 22 clusters
      - leiden_0.55: 23 clusters
      - leiden_0.63: 23 clusters
      - leiden_0.72: 25 clusters
      - leiden_0.8: 26 clusters
    
    Saving processed AnnData object to: /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Emx1_Mut_adult_1/Emx1_Mut_processed.h5ad


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



    
![png](Emx1_Mut_files/Emx1_Mut_18_1.png)
    



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
      <td>6</td>
      <td>0.035765</td>
      <td>-2.055579</td>
      <td>1.000000</td>
      <td>0.372804</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.13</td>
      <td>12</td>
      <td>0.100173</td>
      <td>-1.736170</td>
      <td>1.000000</td>
      <td>0.729083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.22</td>
      <td>14</td>
      <td>0.114606</td>
      <td>-1.611913</td>
      <td>1.000000</td>
      <td>0.779403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.30</td>
      <td>19</td>
      <td>0.120371</td>
      <td>-1.555842</td>
      <td>1.000000</td>
      <td>0.832956</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.38</td>
      <td>21</td>
      <td>0.129418</td>
      <td>-1.499901</td>
      <td>1.000000</td>
      <td>0.865535</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.47</td>
      <td>22</td>
      <td>0.140208</td>
      <td>-1.481119</td>
      <td>1.000000</td>
      <td>0.868788</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.55</td>
      <td>23</td>
      <td>0.134296</td>
      <td>-1.549319</td>
      <td>1.000000</td>
      <td>0.684035</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.63</td>
      <td>23</td>
      <td>0.135538</td>
      <td>-1.555485</td>
      <td>1.000000</td>
      <td>0.682119</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.72</td>
      <td>25</td>
      <td>0.122544</td>
      <td>-1.582776</td>
      <td>0.666667</td>
      <td>0.409560</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.80</td>
      <td>26</td>
      <td>0.123912</td>
      <td>-1.630985</td>
      <td>0.666667</td>
      <td>0.399120</td>
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



    
![png](Emx1_Mut_files/Emx1_Mut_20_1.png)
    



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



    
![png](Emx1_Mut_files/Emx1_Mut_21_1.png)
    


    Individual metrics across resolutions:



    
![png](Emx1_Mut_files/Emx1_Mut_21_3.png)
    



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
      <td>6</td>
      <td>0.372804</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.220000</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.009122</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.13</td>
      <td>12</td>
      <td>0.729083</td>
      <td>0.215840</td>
      <td>0.122323</td>
      <td>0.057706</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.078170</td>
      <td>0.027331</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.22</td>
      <td>14</td>
      <td>0.779403</td>
      <td>0.264205</td>
      <td>0.169910</td>
      <td>0.094225</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.071679</td>
      <td>0.029228</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.30</td>
      <td>19</td>
      <td>0.832956</td>
      <td>0.283525</td>
      <td>0.191383</td>
      <td>0.024370</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.080000</td>
      <td>0.040844</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.38</td>
      <td>21</td>
      <td>0.865535</td>
      <td>0.313843</td>
      <td>0.212807</td>
      <td>0.061546</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.072388</td>
      <td>0.043839</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.47</td>
      <td>22</td>
      <td>0.868788</td>
      <td>0.350000</td>
      <td>0.220000</td>
      <td>0.049764</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.068899</td>
      <td>0.043480</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.55</td>
      <td>23</td>
      <td>0.684035</td>
      <td>0.330190</td>
      <td>0.193881</td>
      <td>0.017559</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.006189</td>
      <td>0.045236</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.63</td>
      <td>23</td>
      <td>0.682119</td>
      <td>0.334350</td>
      <td>0.191520</td>
      <td>0.037477</td>
      <td>0.0</td>
      <td>0.08</td>
      <td>0.001171</td>
      <td>0.045452</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.72</td>
      <td>25</td>
      <td>0.409560</td>
      <td>0.290808</td>
      <td>0.181068</td>
      <td>0.005774</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.80</td>
      <td>26</td>
      <td>0.399120</td>
      <td>0.295389</td>
      <td>0.162606</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.003408</td>
      <td>0.047523</td>
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
      1. Dpy30 (log2FC: 1.51, adj.p-val: 9.91e-46)
      2. Slc16a1 (log2FC: 0.39, adj.p-val: 1.65e-41)
      3. Scamp3 (log2FC: -0.05, adj.p-val: 1.77e-38)
      4. Ubald1 (log2FC: 4.71, adj.p-val: 8.13e-38)
      5. Cby1 (log2FC: 0.63, adj.p-val: 2.88e-36)
      6. Commd4 (log2FC: -0.79, adj.p-val: 3.34e-36)
      7. Slc30a1 (log2FC: 1.31, adj.p-val: 2.93e-35)
      8. Tspyl5 (log2FC: 12.69, adj.p-val: 1.46e-34)
      9. Smim8 (log2FC: -0.2, adj.p-val: 9.28e-34)
      10. 2310009B15Rik (log2FC: -0.92, adj.p-val: 1.91e-33)
    
    Cluster 1:
      1. Ifi47 (log2FC: -2.36, adj.p-val: 9.92e-01)
      2. Gm57230 (log2FC: -3.25, adj.p-val: 9.92e-01)
      3. H2-Q6 (log2FC: -2.59, adj.p-val: 9.92e-01)
      4. Gm40761 (log2FC: -2.31, adj.p-val: 9.92e-01)
      5. Cox8b (log2FC: -3.25, adj.p-val: 9.92e-01)
      6. Iglc3 (log2FC: -1.9, adj.p-val: 9.92e-01)
      7. Gm44109 (log2FC: -1.87, adj.p-val: 9.92e-01)
      8. Kcne1l (log2FC: -2.63, adj.p-val: 9.92e-01)
      9. Spata18 (log2FC: -2.09, adj.p-val: 9.92e-01)
      10. Bglap2 (log2FC: -2.69, adj.p-val: 9.92e-01)
    
    Cluster 2:
      1. Pin4 (log2FC: -0.08, adj.p-val: 1.00e+00)
      2. Hba-a2 (log2FC: 1.75, adj.p-val: 1.00e+00)
      3. Impdh2 (log2FC: -1.73, adj.p-val: 1.00e+00)
      4. Gkn3 (log2FC: -3.77, adj.p-val: 1.00e+00)
      5. Rflnb (log2FC: 0.41, adj.p-val: 1.00e+00)
      6. Fgfbp1 (log2FC: -6.02, adj.p-val: 1.00e+00)
      7. Gm43612 (log2FC: -4.61, adj.p-val: 1.00e+00)
      8. Hba-a1 (log2FC: 4.0, adj.p-val: 1.00e+00)
      9. Or52d13 (log2FC: -4.24, adj.p-val: 1.00e+00)
      10. Gm11748 (log2FC: -6.3, adj.p-val: 1.00e+00)
    
    Cluster 3:
      1. Rab43 (log2FC: 6.95, adj.p-val: 1.18e-01)
      2. Rcc2 (log2FC: 6.87, adj.p-val: 1.56e-01)
      3. Zbtb43 (log2FC: 2.29, adj.p-val: 1.71e-01)
      4. Slc25a11 (log2FC: 3.09, adj.p-val: 1.74e-01)
      5. Spryd7 (log2FC: -3.15, adj.p-val: 1.81e-01)
      6. Insyn1 (log2FC: 3.99, adj.p-val: 1.99e-01)
      7. Mrpl24 (log2FC: 1.32, adj.p-val: 2.27e-01)
      8. Slc39a3 (log2FC: 0.4, adj.p-val: 2.58e-01)
      9. Alg10b (log2FC: 2.26, adj.p-val: 2.67e-01)
      10. Cdk2ap1 (log2FC: -2.17, adj.p-val: 2.88e-01)
    
    Cluster 4:
      1. Hba-a1 (log2FC: 0.17, adj.p-val: 9.99e-01)
      2. Gm16095 (log2FC: -4.24, adj.p-val: 9.99e-01)
      3. Hsd3b5 (log2FC: -5.78, adj.p-val: 9.99e-01)
      4. Aadacl4fm4 (log2FC: -5.04, adj.p-val: 9.99e-01)
      5. Gm15384 (log2FC: -5.29, adj.p-val: 9.99e-01)
      6. Gm38397 (log2FC: -7.28, adj.p-val: 9.99e-01)
      7. Prss40 (log2FC: -8.1, adj.p-val: 9.99e-01)
      8. Crct1 (log2FC: -5.48, adj.p-val: 9.99e-01)
      9. 4930451E10Rik (log2FC: -3.47, adj.p-val: 9.99e-01)
      10. C86187 (log2FC: -4.26, adj.p-val: 9.99e-01)
    
    Cluster 5:
      1. Aprt (log2FC: 2.24, adj.p-val: 1.00e+00)
      2. H4c4 (log2FC: 3.8, adj.p-val: 1.00e+00)
      3. Zfas1 (log2FC: 1.42, adj.p-val: 1.00e+00)
      4. Fam177a (log2FC: -0.99, adj.p-val: 1.00e+00)
      5. Gcsh (log2FC: 2.37, adj.p-val: 1.00e+00)
      6. Abcc4 (log2FC: 3.21, adj.p-val: 1.00e+00)
      7. Ecrg4 (log2FC: -1.78, adj.p-val: 1.00e+00)
      8. Fbll1 (log2FC: 0.54, adj.p-val: 1.00e+00)
      9. Cetn2 (log2FC: 10.11, adj.p-val: 1.00e+00)
      10. S100a9 (log2FC: -0.63, adj.p-val: 1.00e+00)
    
    Cluster 6:
      1. Gm11532 (log2FC: -4.62, adj.p-val: 9.99e-01)
      2. Asmt (log2FC: -5.22, adj.p-val: 9.99e-01)
      3. Gm4791 (log2FC: -6.19, adj.p-val: 9.99e-01)
      4. Fgfbp1 (log2FC: -6.02, adj.p-val: 9.99e-01)
      5. Or2t6 (log2FC: -3.36, adj.p-val: 9.99e-01)
      6. Gm28499 (log2FC: -4.79, adj.p-val: 9.99e-01)
      7. Epp13 (log2FC: -4.04, adj.p-val: 9.99e-01)
      8. Gm12256 (log2FC: -3.34, adj.p-val: 9.99e-01)
      9. Actg2 (log2FC: -3.14, adj.p-val: 9.99e-01)
      10. Gm50057 (log2FC: -5.43, adj.p-val: 9.99e-01)
    
    Cluster 7:
      1. Tmem176b (log2FC: 4.34, adj.p-val: 9.99e-01)
      2. Oasl2 (log2FC: -2.48, adj.p-val: 9.99e-01)
      3. Slc22a18 (log2FC: -2.63, adj.p-val: 9.99e-01)
      4. Tmem37 (log2FC: -3.95, adj.p-val: 9.99e-01)
      5. Mia (log2FC: -3.25, adj.p-val: 9.99e-01)
      6. Ifitm2 (log2FC: -1.46, adj.p-val: 9.99e-01)
      7. 9630013K17Rik (log2FC: -4.74, adj.p-val: 9.99e-01)
      8. Eif3j2 (log2FC: -1.13, adj.p-val: 9.99e-01)
      9. Mip (log2FC: -7.35, adj.p-val: 9.99e-01)
      10. Gm32764 (log2FC: -5.96, adj.p-val: 9.99e-01)
    
    Cluster 8:
      1. Myl9 (log2FC: -1.58, adj.p-val: 1.00e+00)
      2. Dthd1 (log2FC: -2.52, adj.p-val: 1.00e+00)
      3. Gm41654 (log2FC: -9.18, adj.p-val: 1.00e+00)
      4. Gm15339 (log2FC: -3.65, adj.p-val: 1.00e+00)
      5. Ahnak (log2FC: 2.3, adj.p-val: 1.00e+00)
      6. Gm15416 (log2FC: -4.99, adj.p-val: 1.00e+00)
      7. Gm28818 (log2FC: -9.55, adj.p-val: 1.00e+00)
      8. Actg2 (log2FC: -7.43, adj.p-val: 1.00e+00)
      9. Gm40646 (log2FC: -3.81, adj.p-val: 1.00e+00)
      10. Sdc1 (log2FC: -3.46, adj.p-val: 1.00e+00)
    
    Cluster 9:
      1. Cfap68 (log2FC: 5.04, adj.p-val: 1.00e+00)
      2. Mrpl36 (log2FC: 4.51, adj.p-val: 1.00e+00)
      3. Psmb10 (log2FC: 3.34, adj.p-val: 1.00e+00)
      4. Xlr5a (log2FC: 3.74, adj.p-val: 1.00e+00)
      5. Mea1 (log2FC: 3.83, adj.p-val: 1.00e+00)
      6. Cavin3 (log2FC: 0.63, adj.p-val: 1.00e+00)
      7. Atp13a5 (log2FC: 1.08, adj.p-val: 1.00e+00)
      8. H3c11 (log2FC: -2.28, adj.p-val: 1.00e+00)
      9. Pigbos1 (log2FC: 2.48, adj.p-val: 1.00e+00)
      10. Hccs (log2FC: 2.1, adj.p-val: 1.00e+00)
    
    Cluster 10:
      1. Kcne1l (log2FC: -0.49, adj.p-val: 9.96e-01)
      2. Gm56859 (log2FC: -1.84, adj.p-val: 9.96e-01)
      3. Gm56806 (log2FC: -1.16, adj.p-val: 9.96e-01)
      4. Gm2366 (log2FC: -2.41, adj.p-val: 9.96e-01)
      5. Crocc2 (log2FC: 0.17, adj.p-val: 9.96e-01)
      6. 4930563E18Rik (log2FC: -6.12, adj.p-val: 9.96e-01)
      7. Or6z6 (log2FC: -1.57, adj.p-val: 9.96e-01)
      8. Lrrc74b (log2FC: -1.34, adj.p-val: 9.96e-01)
      9. Mecom (log2FC: -1.81, adj.p-val: 9.96e-01)
      10. Gja4 (log2FC: -0.46, adj.p-val: 9.96e-01)
    
    Cluster 11:
      1. Rsad2 (log2FC: -3.94, adj.p-val: 1.00e+00)
      2. 5730414N17Rik (log2FC: 0.23, adj.p-val: 1.00e+00)
      3. S100a8 (log2FC: -0.8, adj.p-val: 1.00e+00)
      4. Igkc (log2FC: -1.14, adj.p-val: 1.00e+00)
      5. Fgfbp1 (log2FC: -1.96, adj.p-val: 1.00e+00)
      6. Gstt3 (log2FC: -1.19, adj.p-val: 1.00e+00)
      7. Apln (log2FC: -0.52, adj.p-val: 1.00e+00)
      8. Gm33936 (log2FC: -0.87, adj.p-val: 1.00e+00)
      9. 2810433D01Rik (log2FC: -5.98, adj.p-val: 1.00e+00)
      10. Gm26887 (log2FC: -4.35, adj.p-val: 1.00e+00)
    
    Cluster 12:
      1. Ly6c1 (log2FC: -2.21, adj.p-val: 9.99e-01)
      2. Hjv (log2FC: 0.2, adj.p-val: 9.99e-01)
      3. Alox15 (log2FC: 0.2, adj.p-val: 9.99e-01)
      4. Nodal (log2FC: 0.2, adj.p-val: 9.99e-01)
      5. Cd19 (log2FC: 0.2, adj.p-val: 9.99e-01)
      6. Gm40915 (log2FC: 0.2, adj.p-val: 9.99e-01)
      7. Gm36462 (log2FC: 0.2, adj.p-val: 9.99e-01)
      8. Prr29 (log2FC: 0.2, adj.p-val: 9.99e-01)
      9. Ccdc185 (log2FC: 0.2, adj.p-val: 9.99e-01)
      10. Gm49466 (log2FC: 0.2, adj.p-val: 9.99e-01)
    
    Cluster 13:
      1. Mrps17 (log2FC: -0.04, adj.p-val: 1.00e+00)
      2. A2ml1 (log2FC: 1.16, adj.p-val: 1.00e+00)
      3. Siva1 (log2FC: -2.04, adj.p-val: 1.00e+00)
      4. Gkn3 (log2FC: -1.0, adj.p-val: 1.00e+00)
      5. AW112010 (log2FC: -4.03, adj.p-val: 1.00e+00)
      6. Bgn (log2FC: -0.03, adj.p-val: 1.00e+00)
      7. Sdhaf4 (log2FC: -1.38, adj.p-val: 1.00e+00)
      8. Gm42620 (log2FC: -3.82, adj.p-val: 1.00e+00)
      9. Eif3j2 (log2FC: -0.6, adj.p-val: 1.00e+00)
      10. Rras (log2FC: -3.38, adj.p-val: 1.00e+00)
    
    Cluster 14:
      1. Tpsg1 (log2FC: 0.2, adj.p-val: 9.98e-01)
      2. Gm13773 (log2FC: 0.2, adj.p-val: 9.98e-01)
      3. 8430437L04Rik (log2FC: 0.2, adj.p-val: 9.98e-01)
      4. Gm29084 (log2FC: 0.2, adj.p-val: 9.98e-01)
      5. Afp (log2FC: 0.2, adj.p-val: 9.98e-01)
      6. Gm10113 (log2FC: 0.2, adj.p-val: 9.98e-01)
      7. Prr29 (log2FC: 0.2, adj.p-val: 9.98e-01)
      8. Or52n4 (log2FC: 0.2, adj.p-val: 9.98e-01)
      9. Cd19 (log2FC: 0.2, adj.p-val: 9.98e-01)
      10. Gm49490 (log2FC: 0.2, adj.p-val: 9.98e-01)
    
    Cluster 15:
      1. Alox15 (log2FC: 0.2, adj.p-val: 9.99e-01)
      2. Hjv (log2FC: 0.2, adj.p-val: 9.99e-01)
      3. Gm36462 (log2FC: 0.2, adj.p-val: 9.99e-01)
      4. Afp (log2FC: 0.2, adj.p-val: 9.99e-01)
      5. Gm29084 (log2FC: 0.2, adj.p-val: 9.99e-01)
      6. Ccdc185 (log2FC: 0.2, adj.p-val: 9.99e-01)
      7. Gm56633 (log2FC: 0.2, adj.p-val: 9.99e-01)
      8. Gm13773 (log2FC: 0.2, adj.p-val: 9.99e-01)
      9. Tpsg1 (log2FC: 0.2, adj.p-val: 9.99e-01)
      10. Gm49466 (log2FC: 0.2, adj.p-val: 9.99e-01)
    
    Cluster 16:
      1. Prg4 (log2FC: -1.73, adj.p-val: 9.99e-01)
      2. Gng11 (log2FC: -0.02, adj.p-val: 9.99e-01)
      3. Bcam (log2FC: -1.17, adj.p-val: 9.99e-01)
      4. Gm36462 (log2FC: 0.2, adj.p-val: 9.99e-01)
      5. Gm29084 (log2FC: 0.2, adj.p-val: 9.99e-01)
      6. Hjv (log2FC: 0.2, adj.p-val: 9.99e-01)
      7. Cd19 (log2FC: 0.2, adj.p-val: 9.99e-01)
      8. Gm49490 (log2FC: 0.2, adj.p-val: 9.99e-01)
      9. Alox15 (log2FC: 0.2, adj.p-val: 9.99e-01)
      10. 8430437L04Rik (log2FC: 0.2, adj.p-val: 9.99e-01)
    
    Cluster 17:
      1. Plscr2 (log2FC: 0.09, adj.p-val: 9.98e-01)
      2. Gm56961 (log2FC: -1.58, adj.p-val: 9.98e-01)
      3. Gm56768 (log2FC: -0.5, adj.p-val: 9.98e-01)
      4. C1qtnf2 (log2FC: -0.84, adj.p-val: 9.98e-01)
      5. Fmod (log2FC: 0.6, adj.p-val: 9.98e-01)
      6. Tmem220 (log2FC: -1.02, adj.p-val: 9.98e-01)
      7. Aspg (log2FC: -1.23, adj.p-val: 9.98e-01)
      8. Gm57009 (log2FC: -0.12, adj.p-val: 9.98e-01)
      9. Apln (log2FC: 0.23, adj.p-val: 9.98e-01)
      10. Gm28172 (log2FC: -0.7, adj.p-val: 9.98e-01)
    
    Cluster 18:
      1. Gm29084 (log2FC: 0.2, adj.p-val: 9.99e-01)
      2. Gm10113 (log2FC: 0.2, adj.p-val: 9.99e-01)
      3. Hjv (log2FC: 0.2, adj.p-val: 9.99e-01)
      4. Gm40915 (log2FC: 0.2, adj.p-val: 9.99e-01)
      5. Ccdc185 (log2FC: 0.2, adj.p-val: 9.99e-01)
      6. Gm56633 (log2FC: 0.2, adj.p-val: 9.99e-01)
      7. Gm36462 (log2FC: 0.2, adj.p-val: 9.99e-01)
      8. Tpsg1 (log2FC: 0.2, adj.p-val: 9.99e-01)
      9. Gm13773 (log2FC: 0.2, adj.p-val: 9.99e-01)
      10. Cd19 (log2FC: 0.2, adj.p-val: 9.99e-01)
    
    Cluster 19:
      1. Gm10113 (log2FC: 0.2, adj.p-val: 9.99e-01)
      2. Gm49466 (log2FC: 0.2, adj.p-val: 9.99e-01)
      3. Gm29084 (log2FC: 0.2, adj.p-val: 9.99e-01)
      4. Hjv (log2FC: 0.2, adj.p-val: 9.99e-01)
      5. Ccdc185 (log2FC: 0.2, adj.p-val: 9.99e-01)
      6. Afp (log2FC: 0.2, adj.p-val: 9.99e-01)
      7. Gm13773 (log2FC: 0.2, adj.p-val: 9.99e-01)
      8. Gm40915 (log2FC: 0.2, adj.p-val: 9.99e-01)
      9. Or52n4 (log2FC: 0.2, adj.p-val: 9.99e-01)
      10. Cd19 (log2FC: 0.2, adj.p-val: 9.99e-01)
    
    Cluster 20:
      1. Slco1a4 (log2FC: 1.69, adj.p-val: 9.99e-01)
      2. Gm49466 (log2FC: 0.2, adj.p-val: 9.99e-01)
      3. Gm29084 (log2FC: 0.2, adj.p-val: 9.99e-01)
      4. Cd19 (log2FC: 0.2, adj.p-val: 9.99e-01)
      5. Afp (log2FC: 0.2, adj.p-val: 9.99e-01)
      6. Gm36462 (log2FC: 0.2, adj.p-val: 9.99e-01)
      7. 8430437L04Rik (log2FC: 0.2, adj.p-val: 9.99e-01)
      8. Gm49490 (log2FC: 0.2, adj.p-val: 9.99e-01)
      9. Tpsg1 (log2FC: 0.2, adj.p-val: 9.99e-01)
      10. Or52n4 (log2FC: 0.2, adj.p-val: 9.99e-01)
    
    Cluster 21:
      1. Gm40915 (log2FC: 0.2, adj.p-val: 9.99e-01)
      2. Alox15 (log2FC: 0.2, adj.p-val: 9.99e-01)
      3. Cd19 (log2FC: 0.2, adj.p-val: 9.99e-01)
      4. Gm49466 (log2FC: 0.2, adj.p-val: 9.99e-01)
      5. Nodal (log2FC: 0.2, adj.p-val: 9.99e-01)
      6. 8430437L04Rik (log2FC: 0.2, adj.p-val: 9.99e-01)
      7. Or52n4 (log2FC: 0.2, adj.p-val: 9.99e-01)
      8. Prr29 (log2FC: 0.2, adj.p-val: 9.99e-01)
      9. Hjv (log2FC: 0.2, adj.p-val: 9.99e-01)
      10. Gm56633 (log2FC: 0.2, adj.p-val: 9.99e-01)



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

Date: 2025-03-28 10:02:26
Analysis duration: 478.4 seconds

Resolutions tested: 10
Resolution range: 0.05 to 0.8

OPTIMAL CLUSTERING RESULT:
- Optimal resolution: 0.47
- Number of clusters: 22
- Silhouette score: 0.1402
- Davies-Bouldin score: 1.4811 (lower is better)
- Calinski-Harabasz score: 332.4
- Avg. marker genes per cluster: 10.0
- Marker gene score: 1.0000
- Overall quality score: 0.8688

Results saved to:
- /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Emx1_Mut_adult_1/my_cluster_analysis

All clustering resolutions saved in the AnnData object:
- leiden_0.05: 6 clusters
- leiden_0.13: 12 clusters
- leiden_0.22: 14 clusters
- leiden_0.3: 19 clusters
- leiden_0.38: 21 clusters
- leiden_0.47: 22 clusters
- leiden_0.55: 23 clusters
- leiden_0.63: 23 clusters
- leiden_0.72: 25 clusters
- leiden_0.8: 26 clusters

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
    Sample: Emx1_Mut
    Optimal resolution: 0.47
    Number of clusters: 22
    Total cells analyzed: 6242
    Results saved to: /beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/individual_data_analysis_opt_clusters/cellranger_counts_R26_Emx1_Mut_adult_1
    ==================================================

