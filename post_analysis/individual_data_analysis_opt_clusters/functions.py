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

def identify_and_visualize_markers(adata, resolutions, output_dir=None, 
                                n_genes=15, n_top_markers=10, n_heatmap_genes=5,
                                leiden_key='leiden', method='wilcoxon', 
                                save_figures=True, show_figures=False):
    """
    Identifies and visualizes marker genes for different clustering resolutions.
    
    This function performs differential expression analysis to find marker genes for each cluster
    at various resolutions, generates ranking plots, saves marker genes to CSV files,
    and creates heatmaps to visualize the expression of top markers. It also generates a summary
    comparison of the number of DEGs across resolutions.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with observations and variables.
    resolutions : list
        List of resolution values to evaluate.
    output_dir : str, optional
        Directory to save outputs. If None, creates 'marker_analysis'.
    n_genes : int, default=15
        Number of top genes to display in the ranking plots.
    n_top_markers : int, default=10
        Number of top markers to save per cluster.
    n_heatmap_genes : int, default=5
        Number of top marker genes per cluster to include in the heatmap.
    leiden_key : str, default='leiden'
        Base name for leiden resolution keys in adata.obs.
    method : str, default='wilcoxon'
        Method for differential expression analysis (e.g., 'wilcoxon', 't-test').
    save_figures : bool, default=True
        Whether to save generated figures to files.
    show_figures : bool, default=False
        Whether to display generated figures.
        
    Returns:
    --------
    dict
        Dictionary containing marker gene dataframes for each resolution.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = "marker_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results
    all_markers = {}
    
    # Process each resolution
    for resolution in tqdm(resolutions, desc="Processing resolutions"):
        # Create resolution-specific key
        leiden_resolution = f"{leiden_key}_{resolution}"
        
        # Skip if clustering hasn't been performed
        if leiden_resolution not in adata.obs:
            print(f"Warning: {leiden_resolution} not found in adata.obs. Skipping.")
            continue
            
        print(f"\nAnalyzing resolution {resolution}:")
        
        # Perform DE analysis
        try:
            # Perform differential expression analysis using scanpy's rank_genes_groups function
            sc.tl.rank_genes_groups(adata, leiden_resolution, method=method, 
                                use_raw=False, key_added=f"rank_genes_{resolution}")
            print(f"  ✓ Identified differentially expressed genes")
        except Exception as e:
            print(f"  ✗ Error in differential expression: {e}")
            continue
        
        # Plot top marker genes
        try:
            plt.figure(figsize=(15, 10))
            # Generate a ranking plot of marker genes using scanpy's rank_genes_groups plotting function
            sc.pl.rank_genes_groups(adata, n_genes=n_genes, sharey=False, 
                                   key=f"rank_genes_{resolution}", show=False)
            plt.suptitle(f"Top marker genes (resolution = {resolution})", fontsize=16)
            plt.tight_layout()
            
            if save_figures:
                plt.savefig(os.path.join(output_dir, f"markers_ranking_res{resolution}.png"), dpi=150)
            
            if show_figures:
                plt.show()
            else:
                plt.close()
                
            print(f"  ✓ Generated marker ranking plot")
        except Exception as e:
            print(f"  ✗ Error generating ranking plot: {e}")
        
        # Get the top markers for each cluster
        try:
            marker_genes = pd.DataFrame()
            n_clusters = len(np.unique(adata.obs[leiden_resolution]))
            
            for i in range(n_clusters):
                # Extract marker genes for each cluster using scanpy's get.rank_genes_groups_df function
                markers = sc.get.rank_genes_groups_df(adata, group=str(i), 
                                                    key=f"rank_genes_{resolution}", log2fc_max=1000)
                markers['cluster'] = i
                marker_genes = pd.concat([marker_genes, markers.head(n_top_markers)])
            
            # Save markers to CSV
            markers_file = os.path.join(output_dir, f"cluster_markers_res{resolution}.csv")
            marker_genes.to_csv(markers_file, index=False)
            print(f"  ✓ Saved {n_top_markers} markers per cluster to {markers_file}")
            
            # Store in results dictionary
            all_markers[resolution] = marker_genes
        except Exception as e:
            print(f"  ✗ Error extracting marker genes: {e}")
            continue
        
        # Generate heatmap of top markers per cluster
        try:
            # Extract top markers per cluster
            top_markers_per_cluster = {}
            for cluster in np.unique(adata.obs[leiden_resolution]):
                cluster_markers = marker_genes[marker_genes['cluster'] == int(cluster)]
                top_markers_per_cluster[cluster] = cluster_markers['names'].tolist()[:n_heatmap_genes]
            
            # Flatten and deduplicate
            markers_flat = [gene for cluster_markers in top_markers_per_cluster.values() 
                           for gene in cluster_markers]
            markers_unique = list(dict.fromkeys(markers_flat))
            
            # Ensure all genes exist in the dataset
            available_markers = [gene for gene in markers_unique if gene in adata.var_names]
            
            if not available_markers:
                print("  ✗ No valid marker genes found in adata.var_names. Heatmap cannot be plotted.")
                continue
                
            if len(available_markers) < len(markers_unique):
                missing = len(markers_unique) - len(available_markers)
                print(f"  ⚠ {missing} marker genes not found in dataset and excluded from heatmap")
            
            # Calculate dendrogram once per resolution
            if f"dendrogram_{leiden_resolution}" not in adata.uns:
                sc.tl.dendrogram(adata, groupby=leiden_resolution)
            
            # Generate heatmap
            plt.figure(figsize=(15, 10))
            # Generate a heatmap of marker gene expression using scanpy's heatmap plotting function
            sc.pl.heatmap(adata, available_markers, groupby=leiden_resolution, 
                         dendrogram=True, swap_axes=True, use_raw=False, 
                         show=False, standard_scale='var', cmap='viridis')
            
            plt.suptitle(f"Marker gene expression heatmap (resolution = {resolution})", fontsize=16)
            plt.tight_layout()
            
            if save_figures:
                plt.savefig(os.path.join(output_dir, f"markers_heatmap_res{resolution}.png"), dpi=150)
            
            if show_figures:
                plt.show()
            else:
                plt.close()
                
            print(f"  ✓ Generated heatmap with {len(available_markers)} genes")
            
        except Exception as e:
            print(f"  ✗ Error generating heatmap: {e}")
    
    # Generate summary comparison
    if all_markers:
        try:
            # Compare number of DEGs across resolutions
            summary_data = []
            for res, markers_df in all_markers.items():
                n_clusters = len(markers_df['cluster'].unique())
                avg_sig_genes = len(markers_df[markers_df['pvals_adj'] < 0.05]) / n_clusters
                summary_data.append({
                    'Resolution': res,
                    'Clusters': n_clusters,
                    'Total_markers': len(markers_df),
                    'Avg_markers_per_cluster': avg_sig_genes
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(output_dir, "resolution_comparison_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"\nSummary comparison saved to {summary_file}")
            
            # Plot comparison metrics
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.bar(summary_df['Resolution'].astype(str), summary_df['Clusters'])
            plt.xlabel('Resolution')
            plt.ylabel('Number of clusters')
            plt.title('Clusters by resolution')
            
            plt.subplot(1, 2, 2)
            plt.bar(summary_df['Resolution'].astype(str), summary_df['Avg_markers_per_cluster'])
            plt.xlabel('Resolution')
            plt.ylabel('Avg. significant markers per cluster')
            plt.title('Marker genes by resolution')
            
            plt.tight_layout()
            
            if save_figures:
                plt.savefig(os.path.join(output_dir, "resolution_comparison.png"), dpi=150)
            
            if show_figures:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            print(f"Error generating summary comparison: {e}")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")
    return all_markers

def evaluate_clustering_quality(adata, resolutions, marker_results=None, leiden_key='leiden',
                               min_genes_per_cluster=10, output_dir=None, show_figures=True):
    """
    Automatically evaluate clustering quality across resolutions to determine 
    the optimal number of clusters.
    
    This function calculates several metrics to assess the quality of clustering at different resolutions,
    including silhouette score, Davies-Bouldin score, Calinski-Harabasz score, and marker gene-based metrics.
    It normalizes these metrics and combines them into an overall score to identify the optimal resolution.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with observations and variables.
    resolutions : list
        List of resolution values to evaluate.
    marker_results : dict, optional
        Dictionary of marker gene dataframes from identify_and_visualize_markers function.
        If None, marker analysis will be performed.
    leiden_key : str, default='leiden'
        Base name for leiden resolution keys in adata.obs.
    min_genes_per_cluster : int, default=10
        Minimum number of significant marker genes required per cluster.
    output_dir : str, optional
        Directory to save outputs. If None, creates 'cluster_evaluation_{timestamp}'.
    show_figures : bool, default=True
        Whether to display figures.
        
    Returns:
    --------
    tuple
        (optimal_resolution, evaluation_df)
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = f"cluster_evaluation"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataframe to store all metrics
    metrics_df = pd.DataFrame(columns=[
        'resolution', 'n_clusters', 'silhouette_score', 'davies_bouldin_score',
        'calinski_harabasz_score', 'avg_marker_genes', 'marker_gene_score',
        'marker_gene_significance', 'cluster_separation_score', 'overall_score'
    ])
    
    # If marker results not provided, calculate them
    if marker_results is None:
        marker_results = {}
        for resolution in tqdm(resolutions, desc="Calculating marker genes"):
            leiden_resolution = f"{leiden_key}_{resolution}"
            if leiden_resolution not in adata.obs:
                print(f"Warning: {leiden_resolution} not found in adata.obs. Skipping.")
                continue
                
            sc.tl.rank_genes_groups(adata, leiden_resolution, method='wilcoxon', 
                                  use_raw=False, key_added=f"rank_genes_{resolution}")
            markers_df = pd.DataFrame()
            for i in range(len(np.unique(adata.obs[leiden_resolution]))):
                try:
                    cluster_markers = sc.get.rank_genes_groups_df(adata, group=str(i), 
                                                           key=f"rank_genes_{resolution}")
                    cluster_markers['cluster'] = i
                    markers_df = pd.concat([markers_df, cluster_markers])
                except:
                    print(f"Warning: Could not get markers for cluster {i} at resolution {resolution}")
            
            marker_results[resolution] = markers_df
    
    print("Evaluating clustering metrics across resolutions...")
    
    # Calculate metrics for each resolution
    for resolution in tqdm(resolutions, desc="Evaluating resolutions"):
        leiden_resolution = f"{leiden_key}_{resolution}"
        
        # Skip if clustering hasn't been performed
        if leiden_resolution not in adata.obs:
            print(f"Warning: {leiden_resolution} not found in adata.obs. Skipping.")
            continue
            
        # Skip if marker results not available
        if resolution not in marker_results:
            print(f"Warning: Marker genes not available for resolution {resolution}. Skipping.")
            continue
        
        # Get cluster labels
        labels = adata.obs[leiden_resolution].astype(int).values
        n_clusters = len(np.unique(labels))
        
        # Get marker genes dataframe
        markers_df = marker_results[resolution]
        
        # Calculate metrics
        metrics_row = {'resolution': resolution, 'n_clusters': n_clusters}
        
        # Default values for metrics in case they fail to compute
        metrics_row['silhouette_score'] = np.nan
        metrics_row['davies_bouldin_score'] = np.nan
        metrics_row['calinski_harabasz_score'] = np.nan
        
        # When calculating metrics for clustering quality
        try:
            max_sample_size = 10000  # Maximum number of cells to use for metrics calculation

            # Use a consistent subsampling across all metrics
            if adata.shape[0] > max_sample_size:
                # Create a single subsample for all metrics
                sample_indices = np.random.choice(adata.shape[0], max_sample_size, replace=False)
                
                # For PCA space if available
                if 'X_pca' in adata.obsm:
                    sample_X_pca = adata.obsm['X_pca'][sample_indices]
                    sample_labels = labels[sample_indices]
                    
                    # Calculate all metrics on the same subsample
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            metrics_row['silhouette_score'] = metrics.silhouette_score(
                                sample_X_pca, sample_labels)
                        except Exception as e:
                            print(f"Warning: Could not compute silhouette score: {e}")
                            
                        try:
                            metrics_row['davies_bouldin_score'] = -metrics.davies_bouldin_score(
                                sample_X_pca, sample_labels)
                        except Exception as e:
                            print(f"Warning: Could not compute Davies-Bouldin score: {e}")
                            
                        try:
                            metrics_row['calinski_harabasz_score'] = metrics.calinski_harabasz_score(
                                sample_X_pca, sample_labels)
                        except Exception as e:
                            print(f"Warning: Could not compute Calinski-Harabasz score: {e}")
                else:
                    # If no PCA, use the same subsample of the original data
                    sample_X = adata.X[sample_indices]
                    sample_labels = labels[sample_indices]
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            metrics_row['silhouette_score'] = metrics.silhouette_score(sample_X, sample_labels)
                        except Exception as e:
                            print(f"Warning: Could not compute silhouette score: {e}")
                            
                        try:
                            metrics_row['davies_bouldin_score'] = -metrics.davies_bouldin_score(sample_X, sample_labels)
                        except Exception as e:
                            print(f"Warning: Could not compute Davies-Bouldin score: {e}")
                            
                        try:
                            metrics_row['calinski_harabasz_score'] = metrics.calinski_harabasz_score(sample_X, sample_labels)
                        except Exception as e:
                            print(f"Warning: Could not compute Calinski-Harabasz score: {e}")
            else:
                # If not subsampling, use full dataset
                if 'X_pca' in adata.obsm:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            metrics_row['silhouette_score'] = metrics.silhouette_score(
                                adata.obsm['X_pca'], labels)
                        except Exception as e:
                            print(f"Warning: Could not compute silhouette score: {e}")
                            
                        try:
                            metrics_row['davies_bouldin_score'] = -metrics.davies_bouldin_score(
                                adata.obsm['X_pca'], labels)
                        except Exception as e:
                            print(f"Warning: Could not compute Davies-Bouldin score: {e}")
                            
                        try:
                            metrics_row['calinski_harabasz_score'] = metrics.calinski_harabasz_score(
                                adata.obsm['X_pca'], labels)
                        except Exception as e:
                            print(f"Warning: Could not compute Calinski-Harabasz score: {e}")
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            metrics_row['silhouette_score'] = metrics.silhouette_score(adata.X, labels)
                        except Exception as e:
                            print(f"Warning: Could not compute silhouette score: {e}")
                            
                        try:
                            metrics_row['davies_bouldin_score'] = -metrics.davies_bouldin_score(adata.X, labels)
                        except Exception as e:
                            print(f"Warning: Could not compute Davies-Bouldin score: {e}")
                            
                        try:
                            metrics_row['calinski_harabasz_score'] = metrics.calinski_harabasz_score(adata.X, labels)
                        except Exception as e:
                            print(f"Warning: Could not compute Calinski-Harabasz score: {e}")
        except Exception as e:
            print(f"Error calculating clustering metrics: {e}")
        
        # 4. Marker gene metrics
        # Average number of significant marker genes per cluster
        sig_markers = markers_df[markers_df['pvals_adj'] < 0.05]
        markers_per_cluster = sig_markers.groupby('cluster').size()
        
        # Skip if no significant markers found
        if len(markers_per_cluster) == 0:
            print(f"Warning: No significant markers found for resolution {resolution}. Skipping.")
            continue
            
        avg_markers = markers_per_cluster.mean()
        metrics_row['avg_marker_genes'] = avg_markers
        
        # Marker gene score: proportion of clusters with sufficient markers
        sufficient_markers = (markers_per_cluster >= min_genes_per_cluster).mean()
        metrics_row['marker_gene_score'] = sufficient_markers
        
        # Marker gene significance: average -log10(p_adj) of top markers
        top_markers = sig_markers.groupby('cluster').head(10)
        if len(top_markers) > 0:
            avg_log_pval = -np.log10(top_markers['pvals_adj'].clip(1e-50)).mean()
            metrics_row['marker_gene_significance'] = avg_log_pval
        else:
            metrics_row['marker_gene_significance'] = 0
        
        # 5. Cluster separation score
        # Log-fold changes between clusters for top marker genes
        if 'logfoldchanges' in markers_df.columns:
            avg_lfc = markers_df.groupby('cluster')['logfoldchanges'].head(10).mean()
            metrics_row['cluster_separation_score'] = avg_lfc if not np.isnan(avg_lfc) else 0
        else:
            metrics_row['cluster_separation_score'] = 0
        
        # Add to dataframe
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)
    
    # Normalize metrics to 0-1 scale for fair comparison
    if len(metrics_df) > 0:
        # For each metric, create a normalized version that's properly scaled
        # Silhouette score: Higher is better (0 to 1)
        metrics_to_normalize = [
            'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score',
            'avg_marker_genes', 'marker_gene_score', 'marker_gene_significance', 
            'cluster_separation_score'
        ]
        
        for metric in metrics_to_normalize:
            # Check if metric has any valid values (not NaN)
            if not metrics_df[metric].isna().all() and metrics_df[metric].std() > 0:
                # Fill NaN values with the worst value
                if metric in ['davies_bouldin_score']:  # Metrics where higher is better (already negated)
                    fill_value = metrics_df[metric].min() if not np.isnan(metrics_df[metric].min()) else 0
                else:  # Metrics where higher is better
                    fill_value = metrics_df[metric].min() if not np.isnan(metrics_df[metric].min()) else 0
                
                # Fill NaN values before normalization
                metrics_df[metric + '_filled'] = metrics_df[metric].fillna(fill_value)
                
                # Normalize
                metrics_df[metric + '_normalized'] = (metrics_df[metric + '_filled'] - metrics_df[metric + '_filled'].min()) / \
                                                   (metrics_df[metric + '_filled'].max() - metrics_df[metric + '_filled'].min())
            else:
                # If all values are NaN or there's no variance, use a neutral value
                metrics_df[metric + '_normalized'] = 0.5
        
        # Calculate overall score with adjusted weights, skipping NaN metrics
        weights = {
            'silhouette_score_normalized': 0.15,
            'davies_bouldin_score_normalized': 0.1,
            'calinski_harabasz_score_normalized': 0.1,
            'marker_gene_score_normalized': 0.25,
            'marker_gene_significance_normalized': 0.2,
            'cluster_separation_score_normalized': 0.2
        }
        
        metrics_df['overall_score'] = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics_df.columns:
                metrics_df['overall_score'] += metrics_df[metric] * weight
                total_weight += weight
        
        # Normalize the overall score if not all metrics were available
        if total_weight > 0 and total_weight < 1.0:
            metrics_df['overall_score'] = metrics_df['overall_score'] / total_weight
            
        # Find optimal resolution
        optimal_idx = metrics_df['overall_score'].idxmax()
        optimal_resolution = metrics_df.loc[optimal_idx, 'resolution']
        optimal_n_clusters = int(metrics_df.loc[optimal_idx, 'n_clusters'])
        
        print(f"\nOptimal clustering resolution: {optimal_resolution}")
        print(f"Optimal number of clusters: {optimal_n_clusters}")
        
        # Save metrics to CSV
        metrics_file = os.path.join(output_dir, "clustering_quality_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Metrics saved to {metrics_file}")
        
        # Plot metrics across resolutions
        plt.figure(figsize=(15, 10))
        
        # Plot normalized metrics
        plt.subplot(2, 2, 1)
        metrics_to_plot = ['silhouette_score_normalized', 'davies_bouldin_score_normalized', 
                          'calinski_harabasz_score_normalized']
        for metric in metrics_to_plot:
            plt.plot(metrics_df['resolution'], metrics_df[metric], 
                    marker='o', label=metric.replace('_normalized', ''))
        plt.axvline(x=optimal_resolution, color='r', linestyle='--', 
                   label=f'Optimal resolution: {optimal_resolution}')
        plt.xlabel('Resolution')
        plt.ylabel('Normalized Score')
        plt.title('Clustering Quality Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot marker gene metrics
        plt.subplot(2, 2, 2)
        metrics_to_plot = ['marker_gene_score_normalized', 'marker_gene_significance_normalized',
                          'cluster_separation_score_normalized']
        for metric in metrics_to_plot:
            plt.plot(metrics_df['resolution'], metrics_df[metric], 
                    marker='o', label=metric.replace('_normalized', ''))
        plt.axvline(x=optimal_resolution, color='r', linestyle='--')
        plt.xlabel('Resolution')
        plt.ylabel('Normalized Score')
        plt.title('Marker Gene Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot overall score
        plt.subplot(2, 2, 3)
        plt.plot(metrics_df['resolution'], metrics_df['overall_score'], 
                marker='o', color='purple', linewidth=2)
        plt.axvline(x=optimal_resolution, color='r', linestyle='--')
        plt.xlabel('Resolution')
        plt.ylabel('Score')
        plt.title('Overall Quality Score')
        plt.grid(True, alpha=0.3)
        
        # Plot number of clusters
        plt.subplot(2, 2, 4)
        plt.plot(metrics_df['resolution'], metrics_df['n_clusters'], 
                marker='s', color='green', linewidth=2)
        plt.axvline(x=optimal_resolution, color='r', linestyle='--')
        plt.xlabel('Resolution')
        plt.ylabel('Number of Clusters')
        plt.title('Cluster Count by Resolution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, "clustering_quality_metrics.png"), dpi=150)
        
        if show_figures:
            plt.show()
        else:
            plt.close()
            
        # Create a detailed report on the optimal clustering
        optimal_leiden = f"{leiden_key}_{optimal_resolution}"
        
        # Visualize optimal clustering
        try:
            # UMAP or tSNE visualization if available
            if 'X_umap' in adata.obsm:
                plt.figure(figsize=(10, 8))
                sc.pl.umap(adata, color=optimal_leiden, legend_loc='on data', 
                          title=f'Optimal Clustering (Resolution={optimal_resolution})', show=False)
                plt.savefig(os.path.join(output_dir, "optimal_clustering_umap.png"), dpi=150)
                if show_figures:
                    plt.show()
                else:
                    plt.close()
            elif 'X_tsne' in adata.obsm:
                plt.figure(figsize=(10, 8))
                sc.pl.tsne(adata, color=optimal_leiden, legend_loc='on data', 
                          title=f'Optimal Clustering (Resolution={optimal_resolution})', show=False)
                plt.savefig(os.path.join(output_dir, "optimal_clustering_tsne.png"), dpi=150)
                if show_figures:
                    plt.show()
                else:
                    plt.close()
        except Exception as e:
            print(f"Error plotting optimal clustering: {e}")
        
        # Generate cluster distinctiveness heatmap
        try:
            if f"rank_genes_{optimal_resolution}" in adata.uns:
                # Get top markers for each cluster
                markers_df = marker_results[optimal_resolution]
                top_markers = {}
                
                for cluster in range(optimal_n_clusters):
                    cluster_markers = markers_df[markers_df['cluster'] == cluster]
                    cluster_markers = cluster_markers.sort_values('pvals_adj')
                    if len(cluster_markers) > 0:
                        top_markers[str(cluster)] = cluster_markers['names'].head(5).tolist()
                
                # Create a dataframe of top markers per cluster
                top_marker_matrix = pd.DataFrame(index=range(optimal_n_clusters))
                
                for cluster, markers in top_markers.items():
                    for i, gene in enumerate(markers):
                        if i < 5:  # Limit to 5 markers per cluster
                            top_marker_matrix.loc[int(cluster), f'Marker {i+1}'] = gene
                
                # Save top markers to CSV
                top_marker_matrix.to_csv(os.path.join(output_dir, "optimal_cluster_top_markers.csv"))
                
                # Plot top markers heatmap
                if 'dendrogram' not in adata.uns or f"{optimal_leiden}" not in adata.uns['dendrogram']:
                    sc.tl.dendrogram(adata, groupby=optimal_leiden)
                
                # Get all unique markers
                all_markers = [gene for markers in top_markers.values() for gene in markers]
                unique_markers = list(dict.fromkeys(all_markers))
                
                # Ensure markers are in the dataset
                available_markers = [gene for gene in unique_markers if gene in adata.var_names]
                
                if available_markers:
                    plt.figure(figsize=(15, 10))
                    sc.pl.heatmap(adata, available_markers, groupby=optimal_leiden, 
                                 dendrogram=True, swap_axes=True, use_raw=False, 
                                 show=False, standard_scale='var', cmap='viridis')
                    plt.suptitle(f'Top Markers for Optimal Clustering (Resolution={optimal_resolution})', 
                                fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "optimal_clustering_heatmap.png"), dpi=150)
                    if show_figures:
                        plt.show()
                    else:
                        plt.close()
        except Exception as e:
            print(f"Error generating optimal clustering heatmap: {e}")
        
        plot_metric_contributions(metrics_df, output_dir)
        
        return optimal_resolution, metrics_df
    else:
        print("No valid evaluations found.")
        return None, metrics_df

def analyze_and_select_best_clustering(adata, resolutions=None, 
                                     run_marker_analysis=True, leiden_key='leiden', 
                                     output_dir=None, show_figures=False):
    """
    Comprehensive analysis of clustering at multiple resolutions and automatic selection
    of the best clustering resolution.
    
    This function orchestrates the entire clustering analysis pipeline, including running Leiden clustering
    at different resolutions, identifying marker genes (optionally), evaluating clustering quality using
    various metrics, and selecting the optimal resolution based on an overall quality score.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with observations and variables.
    resolutions : list, optional
        List of resolution values to evaluate. If None, will use default range.
    run_marker_analysis : bool, default=True
        Whether to run marker gene identification.
    leiden_key : str, default='leiden'
        Base name for leiden resolution keys in adata.obs.
    output_dir : str, optional
        Directory to save outputs.
    show_figures : bool, default=False
        Whether to display figures.
        
    Returns:
    --------
    float
        Optimal resolution
    """
    from datetime import datetime
    import time
    
    # Create output directory
    if output_dir is None:
        output_dir = f"clustering_analysis"
    
    # Set default resolutions if not provided
    if resolutions is None:
        resolutions = np.linspace(0.05, 0.8, 10)
        resolutions = [round(r, 2) for r in resolutions]
    
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    print(f"Analyzing {len(resolutions)} clustering resolutions: {resolutions}")
    
    # Run clustering at each resolution
    print("\nStep 1: Running Leiden clustering at different resolutions...")
    for res in tqdm(resolutions, desc="Computing clusterings"):
        leiden_resolution = f"{leiden_key}_{res}"
        if leiden_resolution not in adata.obs:
            try:
                sc.tl.leiden(adata, resolution=res, key_added=leiden_resolution)
            except Exception as e:
                print(f"Error running leiden clustering at resolution {res}: {e}")
                continue
        # Store the clustering assignments in the main adata object
        # (This is already done by sc.tl.leiden with key_added parameter)
    
    # Run marker gene analysis if requested
    if run_marker_analysis:
        print("\nStep 2: Identifying marker genes for each clustering resolution...")
        marker_dir = os.path.join(output_dir, "marker_analysis")
        os.makedirs(marker_dir, exist_ok=True)

        # Run marker analysis with simplified output
        marker_results = identify_and_visualize_markers(
            adata, resolutions, output_dir=marker_dir, leiden_key=leiden_key,
            n_genes=15, n_top_markers=10, n_heatmap_genes=5,
            method='wilcoxon', save_figures=True, show_figures=show_figures
        )
    else:
        marker_results = None
    
    # Evaluate clustering quality and select best resolution
    print("\nStep 3: Evaluating clustering quality and selecting optimal resolution...")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    optimal_resolution, metrics_df = evaluate_clustering_quality(
        adata, resolutions, marker_results=marker_results,
        leiden_key=leiden_key, output_dir=eval_dir,
        show_figures=show_figures
    )
    
    # Display summary
    if optimal_resolution is not None:
        optimal_leiden = f"{leiden_key}_{optimal_resolution}"
        n_clusters = len(np.unique(adata.obs[optimal_leiden]))
        
        # Save summary report
        elapsed_time = time.time() - start_time
        with open(os.path.join(output_dir, "analysis_summary.txt"), "w") as f:
            f.write("=" * 50 + "\n")
            f.write("CLUSTERING ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis duration: {elapsed_time:.1f} seconds\n\n")
            f.write(f"Resolutions tested: {len(resolutions)}\n")
            f.write(f"Resolution range: {min(resolutions)} to {max(resolutions)}\n\n")
            f.write(f"OPTIMAL CLUSTERING RESULT:\n")
            f.write(f"- Optimal resolution: {optimal_resolution}\n")
            f.write(f"- Number of clusters: {n_clusters}\n")
            
            # Get metrics for optimal resolution
            if not metrics_df.empty:
                opt_metrics = metrics_df[metrics_df['resolution'] == optimal_resolution].iloc[0]
                f.write(f"- Silhouette score: {opt_metrics['silhouette_score']:.4f}\n")
                f.write(f"- Davies-Bouldin score: {-opt_metrics['davies_bouldin_score']:.4f} (lower is better)\n")
                f.write(f"- Calinski-Harabasz score: {opt_metrics['calinski_harabasz_score']:.1f}\n")
                f.write(f"- Avg. marker genes per cluster: {opt_metrics['avg_marker_genes']:.1f}\n")
                f.write(f"- Marker gene score: {opt_metrics['marker_gene_score']:.4f}\n")
                f.write(f"- Overall quality score: {opt_metrics['overall_score']:.4f}\n\n")
            
            f.write("Results saved to:\n")
            f.write(f"- {os.path.abspath(output_dir)}\n")
            
            # Add information about all clusterings saved in the AnnData object
            f.write("\nAll clustering resolutions saved in the AnnData object:\n")
            for res in resolutions:
                res_key = f"{leiden_key}_{res}"
                if res_key in adata.obs:
                    n_clust = len(np.unique(adata.obs[res_key]))
                    f.write(f"- {res_key}: {n_clust} clusters\n")
    
        print(f"\nAnalysis complete in {elapsed_time:.1f} seconds!")
        print(f"Optimal resolution: {optimal_resolution} ({n_clusters} clusters)")
        print(f"All clustering resolutions have been preserved in the AnnData object")
        print(f"Full results saved to {os.path.abspath(output_dir)}")
    
    return optimal_resolution

def plot_metric_contributions(metrics_df, output_dir):
    """
    Create a detailed visualization showing how each metric contributes to the overall score.
    """
    # Create a directory for detailed metrics
    detail_dir = os.path.join(output_dir, "metric_details")
    os.makedirs(detail_dir, exist_ok=True)
    
    # Get all normalized metrics
    normalized_metrics = [col for col in metrics_df.columns if col.endswith('_normalized')]
    
    # Plot individual metric trends across resolutions
    plt.figure(figsize=(15, 12))
    for i, metric in enumerate(normalized_metrics, 1):
        base_metric = metric.replace('_normalized', '')
        plt.subplot(len(normalized_metrics), 1, i)
        plt.plot(metrics_df['resolution'], metrics_df[base_metric], 'o-', label=f'Raw {base_metric}')
        plt.plot(metrics_df['resolution'], metrics_df[metric], 's--', label=f'Normalized {metric}')
        plt.title(f'{base_metric} across resolutions')
        plt.xlabel('Resolution')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(detail_dir, "individual_metrics.png"), dpi=150)
    plt.close()
    
    # Plot stacked contribution of each metric to overall score
    plt.figure(figsize=(12, 8))
    
    # Compute the weighted contribution of each metric
    weights = {
        'silhouette_score_normalized': 0.35,
        'davies_bouldin_score_normalized': 0.22,
        'calinski_harabasz_score_normalized': 0.22,
        'marker_gene_score_normalized': 0.08,
        'marker_gene_significance_normalized': 0.08,
        'cluster_separation_score_normalized': 0.05
    }
    
    # Create a stacked bar chart
    bottom = np.zeros(len(metrics_df))
    for metric in normalized_metrics:
        weight = weights.get(metric, 0)
        contribution = metrics_df[metric] * weight
        plt.bar(metrics_df['resolution'], contribution, bottom=bottom, label=f'{metric} (weight={weight})')
        bottom += contribution
    
    plt.plot(metrics_df['resolution'], metrics_df['overall_score'], 'ro-', linewidth=2, label='Overall score')
    plt.xlabel('Resolution')
    plt.ylabel('Score contribution')
    plt.title('Contribution of each metric to overall score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(detail_dir, "metric_contributions.png"), dpi=150)
    plt.close()
    
    # Create a summary table
    summary = metrics_df[['resolution', 'n_clusters', 'overall_score']]
    for metric in normalized_metrics:
        weight = weights.get(metric, 0)
        summary[f'{metric}_contribution'] = metrics_df[metric] * weight
    
    summary.to_csv(os.path.join(detail_dir, "metric_contribution_summary.csv"), index=False)
    
    print(f"Detailed metric analysis saved to {detail_dir}")