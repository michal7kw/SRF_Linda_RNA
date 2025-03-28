# %% [markdown]
# # Analysis of Unassigned Cells from CellTypist Results
# 
# This notebook analyzes cells that were classified as 'Unassigned' by CellTypist and performs further analysis on these cells to identify potential new cell types or misclassifications.

# %% [markdown]
# ## Environment Setup

# %%
import os
import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %%
# Set working directory - adjust as needed
os.chdir("/beegfs/scratch/ric.broccoli/kubacki.michal/SRF_Linda_RNA/post_analysis/cell_typist/check_unannotated")

# %%
# This cell will be parameterized by the script when creating notebooks
sel_model = "MODEL_PLACEHOLDER"  # This will be replaced with the actual model name
sel_sample = "SAMPLE_PLACEHOLDER"  # This will be replaced with the actual sample name
print(f"Analyzing unassigned cells for model: {sel_model}, sample: {sel_sample}")

# %%
# Configuration parameters
results_dir = f"results_{sel_model}"
output_dir = f"unassigned_analysis_{sel_model}"
min_cells = 10
cluster_resolution = 0.5
n_markers = 20

# Create output directory
os.makedirs(output_dir, exist_ok=True)
sample_outdir = os.path.join(output_dir, sel_sample)
os.makedirs(sample_outdir, exist_ok=True)

print(f"Results will be saved to: {sample_outdir}")

# %% [markdown]
# ## Helper Functions

# %%
def extract_unassigned_cells(adata):
    """Extract cells classified as 'Unassigned' by CellTypist."""
    if 'majority_voting' not in adata.obs.columns:
        raise ValueError("The data does not contain CellTypist annotations (majority_voting column not found)")
    
    # Check if 'Unassigned' is present in the annotations
    if 'Unassigned' not in adata.obs['majority_voting'].values:
        print("No 'Unassigned' cells found in this dataset")
        return None
    
    # Extract unassigned cells
    unassigned_mask = adata.obs['majority_voting'] == 'Unassigned'
    unassigned_count = unassigned_mask.sum()
    print(f"Found {unassigned_count} unassigned cells ({unassigned_count/len(adata.obs)*100:.2f}% of total)")
    
    if unassigned_count > 0:
        return adata[unassigned_mask].copy()
    else:
        return None

def analyze_unassigned_cells(adata_unassigned, cluster_resolution=0.5, n_markers=20):
    """Perform clustering and marker gene analysis on unassigned cells."""
    if adata_unassigned is None or adata_unassigned.n_obs == 0:
        return None
    
    print(f"Analyzing {adata_unassigned.n_obs} unassigned cells")
    
    # Basic preprocessing if not already done
    # Check if normalized data exists
    if 'for_cell_typist' in adata_unassigned.layers:
        adata_unassigned.X = adata_unassigned.layers['for_cell_typist']
    
    # Calculate PCA if not present
    if 'X_pca' not in adata_unassigned.obsm:
        sc.pp.pca(adata_unassigned)
    
    # Calculate neighborhood graph
    sc.pp.neighbors(adata_unassigned)
    
    # Run UMAP for visualization
    sc.tl.umap(adata_unassigned)
    
    # Cluster the unassigned cells to identify potential subpopulations
    sc.tl.leiden(adata_unassigned, resolution=cluster_resolution, key_added='unassigned_clusters')
    
    # Find marker genes for each cluster
    sc.tl.rank_genes_groups(adata_unassigned, 'unassigned_clusters', method='wilcoxon', n_genes=n_markers)
    
    return adata_unassigned

def plot_unassigned_analysis(adata_unassigned, output_dir, sample_name):
    """Generate plots for the unassigned cell analysis."""
    if adata_unassigned is None or adata_unassigned.n_obs == 0:
        return
    
    # Set the output directory for plots
    sc.settings.figdir = output_dir
    
    # Plot UMAP with unassigned clusters
    plt.figure(figsize=(10, 8))
    sc.pl.umap(adata_unassigned, color='unassigned_clusters', 
               title=f"Clustering of Unassigned Cells - {sample_name}",
               save=f"_{sample_name}_unassigned_clusters.png")
    
    # Plot expression of top marker genes for each cluster
    cluster_counts = adata_unassigned.obs['unassigned_clusters'].value_counts()
    print(f"Unassigned clusters and their sizes:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} cells")
    
    # Create a marker gene plot for each cluster
    for cluster in cluster_counts.index:
        try:
            # Get marker genes for this cluster
            marker_genes = adata_unassigned.uns['rank_genes_groups']['names'][cluster][:10]
            
            # Plot the top marker genes for this cluster
            plt.figure(figsize=(12, 5))
            sc.pl.umap(adata_unassigned, color=marker_genes[:5], 
                      title=f"Top markers for cluster {cluster}", 
                      save=f"_{sample_name}_cluster{cluster}_markers.png")
            
            # Create a dot plot for all clusters and their markers
            plt.figure(figsize=(14, 10))
            sc.pl.dotplot(adata_unassigned, marker_genes, groupby='unassigned_clusters',
                          save=f"_{sample_name}_cluster{cluster}_dotplot.png")
        except KeyError:
            print(f"Warning: Could not find marker genes for cluster {cluster}")
    
    # Create a heatmap of marker genes
    try:
        plt.figure(figsize=(14, 10))
        sc.pl.rank_genes_groups_heatmap(adata_unassigned, n_genes=10, 
                                        save=f"_{sample_name}_marker_heatmap.png")
    except KeyError:
        print("Warning: Could not create marker gene heatmap")
    
    # Save the clustered data
    adata_unassigned.write(os.path.join(output_dir, f"{sample_name}_unassigned_analyzed.h5ad"))

def export_marker_genes(adata_unassigned, output_dir, sample_name):
    """Export marker genes for each unassigned cluster to CSV."""
    if adata_unassigned is None or adata_unassigned.n_obs == 0:
        return
    
    try:
        # Get all marker genes information
        marker_genes_dict = {}
        cluster_names = adata_unassigned.obs['unassigned_clusters'].cat.categories
        
        for cluster in cluster_names:
            genes = adata_unassigned.uns['rank_genes_groups']['names'][cluster]
            scores = adata_unassigned.uns['rank_genes_groups']['scores'][cluster]
            pvals = adata_unassigned.uns['rank_genes_groups']['pvals'][cluster]
            pvals_adj = adata_unassigned.uns['rank_genes_groups']['pvals_adj'][cluster]
            
            cluster_markers = pd.DataFrame({
                'gene': genes,
                'score': scores,
                'pval': pvals,
                'pval_adj': pvals_adj
            })
            
            marker_genes_dict[f"cluster_{cluster}"] = cluster_markers
            
            # Save individual cluster markers
            cluster_markers.to_csv(os.path.join(output_dir, f"{sample_name}_cluster{cluster}_markers.csv"), index=False)
        
        # Save a combined marker gene table
        all_markers = pd.concat(marker_genes_dict, names=['cluster', 'rank']).reset_index()
        all_markers.to_csv(os.path.join(output_dir, f"{sample_name}_all_markers.csv"), index=False)
        
    except KeyError as e:
        print(f"Warning: Could not export marker genes: {e}")

def analyze_cell_confidence(adata, adata_unassigned, output_dir, sample_name):
    """Analyze confidence scores for unassigned vs assigned cells."""
    if adata_unassigned is None or adata_unassigned.n_obs == 0:
        return
    
    # Check if confidence scores are available
    if 'conf_score' not in adata.obs.columns:
        print("Warning: Confidence scores not found in the data")
        return
    
    # Compare confidence scores between assigned and unassigned cells
    assigned_mask = adata.obs['majority_voting'] != 'Unassigned'
    assigned_conf = adata.obs.loc[assigned_mask, 'conf_score']
    unassigned_conf = adata.obs.loc[~assigned_mask, 'conf_score']
    
    plt.figure(figsize=(10, 6))
    plt.hist([assigned_conf, unassigned_conf], bins=50, alpha=0.5, 
             label=['Assigned Cells', 'Unassigned Cells'])
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Cells')
    plt.title(f'Confidence Score Distribution - {sample_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{sample_name}_confidence_histogram.png"))
    plt.close()
    
    # Create a boxplot
    plt.figure(figsize=(8, 6))
    conf_data = pd.DataFrame({
        'Confidence': pd.concat([assigned_conf, unassigned_conf]),
        'Group': ['Assigned'] * len(assigned_conf) + ['Unassigned'] * len(unassigned_conf)
    })
    sns.boxplot(x='Group', y='Confidence', data=conf_data)
    plt.title(f'Confidence Score Comparison - {sample_name}')
    plt.savefig(os.path.join(output_dir, f"{sample_name}_confidence_boxplot.png"))
    plt.close()
    
    # Calculate statistics
    conf_stats = pd.DataFrame({
        'Group': ['Assigned', 'Unassigned'],
        'Mean': [assigned_conf.mean(), unassigned_conf.mean()],
        'Median': [assigned_conf.median(), unassigned_conf.median()],
        'Min': [assigned_conf.min(), unassigned_conf.min()],
        'Max': [assigned_conf.max(), unassigned_conf.max()],
        'Count': [len(assigned_conf), len(unassigned_conf)]
    })
    
    conf_stats.to_csv(os.path.join(output_dir, f"{sample_name}_confidence_stats.csv"), index=False)
    print(f"\nConfidence score statistics for {sample_name}:")
    print(conf_stats)

def explore_original_cell_types(adata, output_dir, sample_name):
    """Explore what cell types were assigned in the original data analysis."""
    if 'leiden' in adata.obs.columns:
        # Cross-tabulate original clusters vs CellTypist annotations
        cross_tab = pd.crosstab(adata.obs['leiden'], adata.obs['majority_voting'])
        
        # Save the cross-tabulation
        cross_tab.to_csv(os.path.join(output_dir, f"{sample_name}_leiden_vs_celltypist.csv"))
        
        # Create a heatmap of the cross-tabulation
        plt.figure(figsize=(14, 10))
        sns.heatmap(cross_tab, cmap='viridis', annot=True, fmt='d', cbar=True)
        plt.title(f'Original Clusters vs CellTypist Annotations - {sample_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{sample_name}_leiden_vs_celltypist.png"))
        plt.close()
        
        # Find which clusters have the most unassigned cells
        unassigned_per_cluster = cross_tab['Unassigned'] if 'Unassigned' in cross_tab.columns else pd.Series(0, index=cross_tab.index)
        cluster_sizes = adata.obs['leiden'].value_counts()
        unassigned_pct = (unassigned_per_cluster / cluster_sizes) * 100
        
        # Create a sorted DataFrame
        unassigned_df = pd.DataFrame({
            'Original_Cluster': unassigned_per_cluster.index,
            'Unassigned_Cells': unassigned_per_cluster.values,
            'Total_Cells': [cluster_sizes[c] for c in unassigned_per_cluster.index],
            'Unassigned_Percentage': unassigned_pct.values
        }).sort_values('Unassigned_Percentage', ascending=False)
        
        unassigned_df.to_csv(os.path.join(output_dir, f"{sample_name}_unassigned_by_cluster.csv"), index=False)
        print(f"\nOriginal clusters with most unassigned cells for {sample_name}:")
        print(unassigned_df.head(10))

# %% [markdown]
# ## Load Data
# 
# First, we load the annotated data file created by CellTypist.

# %%
# Path to the annotated h5ad file
annotated_file = os.path.join(results_dir, f"{sel_sample}_annotated.h5ad")

# Check if file exists
if not os.path.exists(annotated_file):
    raise FileNotFoundError(f"Annotated file not found: {annotated_file}")

# Load the data
print(f"Loading data from {annotated_file}")
adata = sc.read_h5ad(annotated_file)
print(f"Loaded data with {adata.n_obs} cells and {adata.n_vars} genes")

# %% [markdown]
# ## Extract and Analyze Unassigned Cells

# %%
# Extract unassigned cells
adata_unassigned = extract_unassigned_cells(adata)

# %%
# If there are enough unassigned cells, analyze them further
if adata_unassigned is not None and adata_unassigned.n_obs >= min_cells:
    print(f"Proceeding with analysis of {adata_unassigned.n_obs} unassigned cells")
    
    # Analyze unassigned cells
    adata_analyzed = analyze_unassigned_cells(
        adata_unassigned, 
        cluster_resolution=cluster_resolution,
        n_markers=n_markers
    )
    
    # Check if analysis was successful
    if adata_analyzed is not None:
        print("Unassigned cell analysis completed successfully")
    else:
        print("Analysis of unassigned cells failed")
else:
    print(f"Skipping further analysis: Not enough unassigned cells")
    adata_analyzed = None

# %% [markdown]
# ## Generate Visualizations and Analyze Results

# %%
# If we have analyzed data, generate plots and export results
if adata_analyzed is not None:
    # Plot UMAP with clusters and marker genes
    plot_unassigned_analysis(adata_analyzed, sample_outdir, sel_sample)
    
    # Export marker genes to CSV
    export_marker_genes(adata_analyzed, sample_outdir, sel_sample)
    
    # Analyze confidence scores
    analyze_cell_confidence(adata, adata_unassigned, sample_outdir, sel_sample)
    
    # Compare with original clustering
    explore_original_cell_types(adata, sample_outdir, sel_sample)

# %% [markdown]
# ## Explore Clusters of Unassigned Cells
# 
# Let's look at each cluster of unassigned cells in more detail.

# %%
# Only run this if we have analyzed unassigned cells
if adata_analyzed is not None:
    # Check the size of each cluster
    cluster_counts = adata_analyzed.obs['unassigned_clusters'].value_counts()
    print("Unassigned cell clusters and their sizes:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} cells ({count/adata_analyzed.n_obs*100:.2f}%)")
    
    # Plot the UMAP of unassigned cells colored by cluster
    sc.pl.umap(adata_analyzed, color='unassigned_clusters', 
               title=f"Clusters of Unassigned Cells - {sel_sample}")

# %%
# Look at top marker genes for each cluster
if adata_analyzed is not None and 'rank_genes_groups' in adata_analyzed.uns:
    for cluster in cluster_counts.index:
        try:
            # Get top 10 marker genes for this cluster
            markers = adata_analyzed.uns['rank_genes_groups']['names'][cluster][:10]
            scores = adata_analyzed.uns['rank_genes_groups']['scores'][cluster][:10]
            
            print(f"\nTop marker genes for cluster {cluster}:")
            for i, (gene, score) in enumerate(zip(markers, scores)):
                print(f"  {i+1}. {gene} (score: {score:.4f})")
                
            # Create a violin plot of the top 5 markers
            sc.pl.violin(adata_analyzed, markers[:5], groupby='unassigned_clusters', 
                        rotation=90, title=f"Marker expression for cluster {cluster}")
        except:
            print(f"Could not find markers for cluster {cluster}")

# %% [markdown]
# ## Compare Unassigned Cells with Known Cell Types
# 
# Here we'll try to understand how the unassigned cells relate to the known cell types in the CellTypist model.

# %%
# Get probability columns from the original data
if adata_unassigned is not None:
    prob_columns = [col for col in adata.obs.columns if col.startswith('prob_')]
    
    # Check the probability distribution for unassigned cells
    if prob_columns:
        # Copy the probability scores to the unassigned cells object
        for col in prob_columns:
            adata_unassigned.obs[col] = adata.obs.loc[adata_unassigned.obs_names, col]
        
        # Debug information
        print(f"Number of probability columns: {len(prob_columns)}")
        for col in prob_columns:
            print(f"Column {col} dtype: {adata_unassigned.obs[col].dtype}")
            # Print sample values
            print(f"Sample values from {col}: {adata_unassigned.obs[col].head(3).values}")
        
        # Try to ensure all columns are numeric
        numeric_prob_columns = []
        for col in prob_columns:
            try:
                adata_unassigned.obs[col] = pd.to_numeric(adata_unassigned.obs[col], errors='coerce')
                numeric_prob_columns.append(col)
            except Exception as e:
                print(f"Could not convert column {col} to numeric: {e}")
        
        if numeric_prob_columns:
            # Calculate the maximum probability for each cell using only numeric columns
            adata_unassigned.obs['max_prob'] = adata_unassigned.obs[numeric_prob_columns].max(axis=1)
            
            # Plot the distribution of maximum probabilities
            plt.figure(figsize=(10, 6))
            plt.hist(adata_unassigned.obs['max_prob'], bins=50)
            plt.xlabel('Maximum Probability Score')
            plt.ylabel('Number of Cells')
            plt.title('Distribution of Maximum Cell Type Probabilities for Unassigned Cells')
            plt.savefig(os.path.join(sample_outdir, f"{sel_sample}_max_probability_hist.png"))
            plt.show()
            
            # Find the closest cell type for each unassigned cell
            closest_celltype = pd.DataFrame(index=adata_unassigned.obs_names)
            closest_celltype['max_prob'] = adata_unassigned.obs['max_prob']
            closest_celltype['closest_type'] = adata_unassigned.obs[numeric_prob_columns].idxmax(axis=1)
            closest_celltype['closest_type'] = closest_celltype['closest_type'].str.replace('prob_', '')
            
            # Count the closest cell types
            celltype_counts = closest_celltype['closest_type'].value_counts()
            
            plt.figure(figsize=(14, 8))
            celltype_counts.plot(kind='bar')
            plt.title('Closest Cell Types for Unassigned Cells')
            plt.xlabel('Cell Type')
            plt.ylabel('Number of Cells')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(sample_outdir, f"{sel_sample}_closest_celltypes.png"))
            plt.show()
            
            # Save this information
            closest_celltype.to_csv(os.path.join(sample_outdir, f"{sel_sample}_closest_celltypes.csv"))
        else:
            print("Warning: No numeric probability columns found. Skipping probability analysis.")

# %% [markdown]
# ## Summary of Findings
# 
# Here we summarize the key findings from the analysis of unassigned cells.

# %%
# Print summary statistics
if adata_unassigned is not None:
    print(f"\n{'='*50}")
    print(f"UNASSIGNED CELL ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Sample: {sel_sample}")
    print(f"Model: {sel_model}")
    print(f"Total cells in dataset: {adata.n_obs}")
    print(f"Unassigned cells: {adata_unassigned.n_obs} ({adata_unassigned.n_obs/adata.n_obs*100:.2f}%)")
    
    if adata_analyzed is not None:
        print(f"Number of clusters found: {len(cluster_counts)}")
        print("\nLargest clusters of unassigned cells:")
        for cluster, count in cluster_counts.iloc[:5].items():
            print(f"  Cluster {cluster}: {count} cells")
        
        if 'closest_type' in locals():
            print("\nMost common closest cell types:")
            for celltype, count in celltype_counts.iloc[:5].items():
                print(f"  {celltype}: {count} cells")
    
    print(f"\nResults saved to: {os.path.abspath(sample_outdir)}")
    print(f"{'='*50}")

# %% [markdown]
# ## Save Final Results

# %%
# Save a copy of the original data with unassigned cluster information
if adata_analyzed is not None:
    # Create a new column in the original data for unassigned clusters
    adata.obs['unassigned_cluster'] = 'Assigned'
    
    # Update with cluster information for unassigned cells
    for cell in adata_analyzed.obs_names:
        adata.obs.loc[cell, 'unassigned_cluster'] = f"Unassigned_{adata_analyzed.obs.loc[cell, 'unassigned_clusters']}"
    
    # Save the updated data
    output_file = os.path.join(sample_outdir, f"{sel_sample}_with_unassigned_clusters.h5ad")
    adata.write(output_file)
    print(f"Saved updated data with unassigned cluster information to {output_file}")

# %%
print(f"\n{'='*50}")
print(f"UNASSIGNED CELL ANALYSIS COMPLETED")
print(f"{'='*50}") 