#!/usr/bin/env python3
"""
This script loads annotated data from CellTypist results, extracts cells classified as 'Unassigned',
and performs further analysis on these cells to identify potential new cell types or misclassifications.

Usage:
    python analyze_unassigned_cells.py --results-dir results_Mouse_Isocortex_Hippocampus --output-dir unassigned_analysis
"""

import os
import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze unassigned cells from CellTypist results')
    parser.add_argument('--results-dir', required=True, help='Directory containing CellTypist results')
    parser.add_argument('--output-dir', default='unassigned_analysis', help='Directory to save analysis results')
    parser.add_argument('--samples', nargs='+', default=['Emx1_Ctrl', 'Emx1_Mut', 'Nestin_Ctrl', 'Nestin_Mut'],
                        help='List of sample names to analyze')
    parser.add_argument('--min-cells', type=int, default=10,
                        help='Minimum number of unassigned cells required for analysis')
    parser.add_argument('--cluster-resolution', type=float, default=0.5,
                        help='Resolution for clustering unassigned cells')
    parser.add_argument('--n-markers', type=int, default=20,
                        help='Number of marker genes to identify for each cluster')
    return parser.parse_args()

def find_annotated_h5ad_files(results_dir, samples):
    """Find all annotated h5ad files in the results directory for the given samples."""
    h5ad_files = {}
    for sample in samples:
        file_path = os.path.join(results_dir, f"{sample}_annotated.h5ad")
        if os.path.exists(file_path):
            h5ad_files[sample] = file_path
        else:
            print(f"Warning: No annotated file found for sample {sample} in {results_dir}")
    return h5ad_files

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
    
    # Create output directory
    sample_outdir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_outdir, exist_ok=True)
    
    # Plot UMAP with unassigned clusters
    sc.settings.figdir = sample_outdir
    
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
    adata_unassigned.write(os.path.join(sample_outdir, f"{sample_name}_unassigned_analyzed.h5ad"))

def export_marker_genes(adata_unassigned, output_dir, sample_name):
    """Export marker genes for each unassigned cluster to CSV."""
    if adata_unassigned is None or adata_unassigned.n_obs == 0:
        return
    
    sample_outdir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_outdir, exist_ok=True)
    
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
            cluster_markers.to_csv(os.path.join(sample_outdir, f"{sample_name}_cluster{cluster}_markers.csv"), index=False)
        
        # Save a combined marker gene table
        all_markers = pd.concat(marker_genes_dict, names=['cluster', 'rank']).reset_index()
        all_markers.to_csv(os.path.join(sample_outdir, f"{sample_name}_all_markers.csv"), index=False)
        
    except KeyError as e:
        print(f"Warning: Could not export marker genes: {e}")

def analyze_cell_confidence(adata, adata_unassigned, output_dir, sample_name):
    """Analyze confidence scores for unassigned vs assigned cells."""
    if adata_unassigned is None or adata_unassigned.n_obs == 0:
        return
    
    sample_outdir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_outdir, exist_ok=True)
    
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
    plt.savefig(os.path.join(sample_outdir, f"{sample_name}_confidence_histogram.png"))
    plt.close()
    
    # Create a boxplot
    plt.figure(figsize=(8, 6))
    conf_data = pd.DataFrame({
        'Confidence': pd.concat([assigned_conf, unassigned_conf]),
        'Group': ['Assigned'] * len(assigned_conf) + ['Unassigned'] * len(unassigned_conf)
    })
    sns.boxplot(x='Group', y='Confidence', data=conf_data)
    plt.title(f'Confidence Score Comparison - {sample_name}')
    plt.savefig(os.path.join(sample_outdir, f"{sample_name}_confidence_boxplot.png"))
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
    
    conf_stats.to_csv(os.path.join(sample_outdir, f"{sample_name}_confidence_stats.csv"), index=False)
    print(f"\nConfidence score statistics for {sample_name}:")
    print(conf_stats)

def explore_original_cell_types(adata, output_dir, sample_name):
    """Explore what cell types were assigned in the original data analysis."""
    if 'leiden' in adata.obs.columns:
        # Create output directory
        sample_outdir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_outdir, exist_ok=True)
        
        # Cross-tabulate original clusters vs CellTypist annotations
        cross_tab = pd.crosstab(adata.obs['leiden'], adata.obs['majority_voting'])
        
        # Save the cross-tabulation
        cross_tab.to_csv(os.path.join(sample_outdir, f"{sample_name}_leiden_vs_celltypist.csv"))
        
        # Create a heatmap of the cross-tabulation
        plt.figure(figsize=(14, 10))
        sns.heatmap(cross_tab, cmap='viridis', annot=True, fmt='d', cbar=True)
        plt.title(f'Original Clusters vs CellTypist Annotations - {sample_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_outdir, f"{sample_name}_leiden_vs_celltypist.png"))
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
        
        unassigned_df.to_csv(os.path.join(sample_outdir, f"{sample_name}_unassigned_by_cluster.csv"), index=False)
        print(f"\nOriginal clusters with most unassigned cells for {sample_name}:")
        print(unassigned_df.head(10))

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find annotated h5ad files
    h5ad_files = find_annotated_h5ad_files(args.results_dir, args.samples)
    
    if not h5ad_files:
        print(f"Error: No annotated h5ad files found in {args.results_dir}")
        return
    
    # Process each sample
    for sample_name, file_path in h5ad_files.items():
        print(f"\n{'='*50}")
        print(f"Processing sample: {sample_name}")
        print(f"{'='*50}")
        
        # Load the data
        try:
            adata = sc.read_h5ad(file_path)
            print(f"Loaded data with {adata.n_obs} cells and {adata.n_vars} genes")
            
            # Extract unassigned cells
            adata_unassigned = extract_unassigned_cells(adata)
            
            if adata_unassigned is not None and adata_unassigned.n_obs >= args.min_cells:
                # Analyze unassigned cells
                adata_analyzed = analyze_unassigned_cells(
                    adata_unassigned, 
                    cluster_resolution=args.cluster_resolution,
                    n_markers=args.n_markers
                )
                
                # Generate plots and save results
                if adata_analyzed is not None:
                    plot_unassigned_analysis(adata_analyzed, args.output_dir, sample_name)
                    export_marker_genes(adata_analyzed, args.output_dir, sample_name)
                    analyze_cell_confidence(adata, adata_unassigned, args.output_dir, sample_name)
                    explore_original_cell_types(adata, args.output_dir, sample_name)
            else:
                print(f"Skipping further analysis: Not enough unassigned cells")
                
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
    
    print("\nUnassigned cell analysis completed!")

if __name__ == "__main__":
    main() 