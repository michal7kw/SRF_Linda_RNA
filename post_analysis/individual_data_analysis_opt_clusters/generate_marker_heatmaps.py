#!/usr/bin/env python3
"""
This script takes saved .h5ad files and generates marker gene heatmaps
for different clustering resolutions and cluster-specific biomarkers.
"""

import os
import sys
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm

def generate_heatmap(adata, resolution, leiden_key='leiden', 
                     output_dir=None, cluster_specific=False, n_genes=5,
                     show_figures=False, custom_markers=None):
    """
    Generate marker gene heatmaps for a specific resolution.
    
    Parameters:
    ----------
    adata : AnnData
        Annotated data matrix with clustering results.
    resolution : float
        Resolution value for the leiden clustering.
    leiden_key : str, default='leiden'
        Base name for the leiden clustering key in adata.obs.
    output_dir : str, optional
        Directory to save the heatmaps.
    cluster_specific : bool, default=False
        Whether to create separate heatmaps for each cluster's markers.
    n_genes : int, default=5
        Number of top marker genes per cluster to include.
    show_figures : bool, default=False
        Whether to display generated figures.
    custom_markers : dict, optional
        Optional dictionary mapping cluster ids to lists of marker genes.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = "heatmaps"
    
    os.makedirs(output_dir, exist_ok=True)
    
    leiden_resolution = f"{leiden_key}_{resolution}"
    
    # Check if the clustering exists
    if leiden_resolution not in adata.obs:
        print(f"Error: {leiden_resolution} not found in the data.")
        return
    
    n_clusters = len(np.unique(adata.obs[leiden_resolution]))
    print(f"Processing resolution {resolution} with {n_clusters} clusters")
    
    # Check if marker gene ranking exists or load from CSV
    if f"rank_genes_{resolution}" not in adata.uns:
        print(f"  Marker gene ranking not found in the data for resolution {resolution}")
        # Try to compute it
        try:
            print(f"  Computing marker genes for resolution {resolution}...")
            sc.tl.rank_genes_groups(adata, leiden_resolution, method='wilcoxon',
                                  use_raw=False, key_added=f"rank_genes_{resolution}")
        except Exception as e:
            print(f"  Error computing marker genes: {e}")
            return
    
    # Calculate dendrogram for this resolution if not already done
    try:
        if 'dendrogram' not in adata.uns or f"{leiden_resolution}" not in adata.uns['dendrogram']:
            sc.tl.dendrogram(adata, groupby=leiden_resolution)
    except Exception as e:
        print(f"  Error calculating dendrogram: {e}")
    
    # Get markers from custom list, otherwise use computed markers
    if custom_markers:
        print(f"  Using custom markers")
        all_markers = []
        for cluster, markers in custom_markers.items():
            all_markers.extend(markers)
        
        # Deduplicate markers
        all_markers = list(dict.fromkeys(all_markers))
        
        # Filter markers to ensure they exist in the dataset
        available_markers = [gene for gene in all_markers if gene in adata.var_names]
        
        if len(available_markers) < len(all_markers):
            print(f"  Warning: {len(all_markers) - len(available_markers)} markers not found in the dataset")
        
        if not available_markers:
            print(f"  Error: No valid markers found in the dataset")
            return
    else:
        # Extract top markers per cluster from the data
        try:
            top_markers_per_cluster = {}
            for cluster in range(n_clusters):
                # Get markers for this cluster
                markers = sc.get.rank_genes_groups_df(adata, group=str(cluster),
                                                    key=f"rank_genes_{resolution}")
                top_genes = markers['names'].head(n_genes).tolist()
                top_markers_per_cluster[str(cluster)] = top_genes
            
            # Flatten and deduplicate markers
            all_markers = [gene for markers in top_markers_per_cluster.values() for gene in markers]
            all_markers = list(dict.fromkeys(all_markers))
            
            # Ensure all genes exist in the dataset
            available_markers = [gene for gene in all_markers if gene in adata.var_names]
            
            if len(available_markers) < len(all_markers):
                print(f"  Warning: {len(all_markers) - len(available_markers)} markers not found in the dataset")
            
            if not available_markers:
                print(f"  Error: No valid markers found in the dataset")
                return
        except Exception as e:
            print(f"  Error extracting markers: {e}")
            return
    
    # Generate combined heatmap with all markers
    try:
        plt.figure(figsize=(16, 12))
        
        sc.pl.heatmap(adata, available_markers, groupby=leiden_resolution,
                     dendrogram=True,
                     swap_axes=True,               # Put genes on y-axis for better labels
                     show_gene_labels=True,        # Show gene names
                     use_raw=False,                # Use normalized data
                     standard_scale='var',         # Scale expression by gene
                     cmap='viridis',               # Better colormap
                     vmin=0,                       # Set minimum value to 0
                     vmax=None,                    # Let max scale automatically
                     show=False)
        
        # Add title without using tight_layout
        plt.suptitle(f"Marker gene expression heatmap (resolution = {resolution})", fontsize=16, y=0.98)
        
        # Use bbox_inches='tight' in savefig instead of tight_layout
        plt.savefig(os.path.join(output_dir, f"markers_heatmap_res{resolution}.png"), 
                   dpi=150, bbox_inches='tight')
        
        if show_figures:
            plt.show()
        else:
            plt.close()
            
        print(f"  Generated heatmap with {len(available_markers)} marker genes")
    except Exception as e:
        print(f"  Error generating heatmap: {e}")
    
    # Generate cluster-specific heatmaps if requested
    if cluster_specific and top_markers_per_cluster:
        # Create a subdirectory for cluster-specific heatmaps
        cluster_dir = os.path.join(output_dir, f"clusters_res{resolution}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        for cluster, markers in top_markers_per_cluster.items():
            # Filter markers to ensure they exist in the dataset
            available_markers = [gene for gene in markers if gene in adata.var_names]
            
            if not available_markers:
                print(f"  Warning: No valid markers for cluster {cluster}")
                continue
                
            try:
                plt.figure(figsize=(14, 8))
                
                # Create a cluster-focused heatmap
                cluster_mask = adata.obs[leiden_resolution] == cluster
                cluster_cells = np.where(cluster_mask)[0]
                other_cells = np.where(~cluster_mask)[0]
                
                # Concatenate indices so cluster cells come first
                cell_order = np.concatenate([cluster_cells, other_cells])
                
                # Create a temporary observation for coloring
                adata.obs['cluster_focus'] = 'Other'
                adata.obs.loc[adata.obs[leiden_resolution] == cluster, 'cluster_focus'] = f'Cluster {cluster}'
                
                # Generate the heatmap
                sc.pl.heatmap(adata, available_markers, 
                             groupby='cluster_focus',
                             dendrogram=False,
                             show_gene_labels=True,
                             use_raw=False,
                             standard_scale='var',
                             cmap='viridis',
                             vmin=0,
                             vmax=None,
                             show=False)
                
                # Use suptitle instead of title and avoid tight_layout
                plt.suptitle(f"Top markers for Cluster {cluster} (resolution = {resolution})", fontsize=16, y=0.98)
                
                # Use bbox_inches='tight' in savefig
                plt.savefig(os.path.join(cluster_dir, f"cluster{cluster}_markers.png"), 
                           dpi=150, bbox_inches='tight')
                
                if show_figures:
                    plt.show()
                else:
                    plt.close()
                
                print(f"  Generated heatmap for cluster {cluster} with {len(available_markers)} markers")
            except Exception as e:
                print(f"  Error generating heatmap for cluster {cluster}: {e}")
    
    print(f"  Heatmap generation complete for resolution {resolution}")

def main():
    parser = argparse.ArgumentParser(description='Generate marker gene heatmaps from saved .h5ad files')
    parser.add_argument('--adata', required=True, help='Path to saved .h5ad file')
    parser.add_argument('--resolutions', nargs='+', type=float, help='List of resolution values to process')
    parser.add_argument('--leiden_key', default='leiden', help='Base name for leiden resolution keys')
    parser.add_argument('--output_dir', default='heatmaps', help='Directory to save heatmaps')
    parser.add_argument('--n_genes', type=int, default=5, help='Number of top marker genes per cluster')
    parser.add_argument('--cluster_specific', action='store_true', help='Generate cluster-specific heatmaps')
    parser.add_argument('--show_figures', action='store_true', help='Display generated figures')
    parser.add_argument('--marker_file', help='Path to custom marker gene CSV file')
    
    args = parser.parse_args()
    
    # Load the AnnData object
    print(f"Loading data from {args.adata}...")
    try:
        adata = sc.read_h5ad(args.adata)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    print(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Load custom markers if provided
    custom_markers = None
    if args.marker_file:
        try:
            marker_df = pd.read_csv(args.marker_file)
            custom_markers = {}
            
            # Determine the format of the marker file
            if 'cluster' in marker_df.columns and 'gene' in marker_df.columns:
                # Standard format with 'cluster' and 'gene' columns
                for _, row in marker_df.iterrows():
                    cluster = str(row['cluster'])
                    gene = row['gene']
                    if cluster not in custom_markers:
                        custom_markers[cluster] = []
                    custom_markers[cluster].append(gene)
            elif 'cluster' in marker_df.columns and 'names' in marker_df.columns:
                # Format from scanpy output
                for _, row in marker_df.iterrows():
                    cluster = str(row['cluster'])
                    gene = row['names']
                    if cluster not in custom_markers:
                        custom_markers[cluster] = []
                    custom_markers[cluster].append(gene)
            else:
                # Assume columns are cluster ids and rows contain genes
                for col in marker_df.columns:
                    if col != 'gene':  # Skip gene name column if it exists
                        custom_markers[col] = marker_df[col].dropna().tolist()
            
            print(f"Loaded custom markers for {len(custom_markers)} clusters")
        except Exception as e:
            print(f"Error loading custom markers: {e}")
            custom_markers = None
    
    # Get available leiden clusterings if resolutions not provided
    if not args.resolutions:
        leiden_columns = [col for col in adata.obs.columns if col.startswith(f"{args.leiden_key}_")]
        resolutions = [float(col.split('_')[1]) for col in leiden_columns]
        resolutions.sort()
        
        if not resolutions:
            print("No leiden clusterings found in the data. Please specify resolutions.")
            sys.exit(1)
            
        print(f"Found {len(resolutions)} leiden clusterings: {resolutions}")
    else:
        resolutions = args.resolutions
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate heatmaps for each resolution
    for resolution in resolutions:
        generate_heatmap(adata, resolution, 
                        leiden_key=args.leiden_key,
                        output_dir=args.output_dir,
                        cluster_specific=args.cluster_specific,
                        n_genes=args.n_genes,
                        show_figures=args.show_figures,
                        custom_markers=custom_markers)
    
    print("Heatmap generation complete!")

if __name__ == "__main__":
    main() 