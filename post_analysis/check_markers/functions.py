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

def plot_marker_gene(adata, selected_cell_type, marker_gene, markers_dict):
    if selected_cell_type in markers_dict and marker_gene in adata.var_names and marker_gene in markers_dict[selected_cell_type]:
        fig, axes = plt.subplots(1, 3, figsize=(36, 10), dpi=80)

        try:
            # Check if 'for_cell_typist' layer exists and use it for visualization
            if 'for_cell_typist' in adata.layers:
                # Create temporary view with normalized data
                temp_adata = adata.copy()
                temp_adata.X = adata.layers['for_cell_typist']
                
                # Plot using the normalized data
                sc.pl.umap(temp_adata, color=marker_gene, ax=axes[0], 
                           title=f"{marker_gene} ({selected_cell_type})", cmap='Reds', show=False,
                           size=150)
            else:
                # Use existing data if normalized layer not available
                sc.pl.umap(adata, color=marker_gene, ax=axes[0], 
                           title=f"{marker_gene} ({selected_cell_type})", cmap='Reds', show=False,
                           size=150)
                
            axes[0].set_xlabel("UMAP1", fontsize=20)
            axes[0].set_ylabel("UMAP2", fontsize=20)
            axes[0].tick_params(axis='both', which='major', labelsize=18)
            axes[0].set_title(f"{marker_gene} ({selected_cell_type})", fontsize=26)

            # Plot leiden_0.38
            sc.pl.umap(adata, color='leiden_0.38', ax=axes[1], 
                        title='leiden_0.38', show=False, size=150, legend_loc = 'none')
            axes[1].set_xlabel("UMAP1", fontsize=20)
            axes[1].set_ylabel("UMAP2", fontsize=20)
            axes[1].tick_params(axis='both', which='major', labelsize=18)
            axes[1].set_title('leiden_0.38', fontsize=26)

            # Plot majority_voting
            sc.pl.umap(adata, color='majority_voting', ax=axes[2], 
                        title='majority_voting', show=False, size=150, legend_loc = 'right margin')
            axes[2].set_xlabel("UMAP1", fontsize=20)
            axes[2].set_ylabel("UMAP2", fontsize=20)
            axes[2].tick_params(axis='both', which='major', labelsize=18)
            axes[2].set_title('majority_voting', fontsize=26)

        except Exception as e:
            print(f"Error plotting: {e}")

        plt.tight_layout()
        plt.show()
    else:
        print(f"Marker gene {marker_gene} not found for cell type {selected_cell_type} or gene not in adata.var_names")


def plot_marker_genes(adata, selected_cell_type, markers_dict):
    all_genes_to_plot = []

    if selected_cell_type in markers_dict:
        all_genes_to_plot = [gene for gene in markers_dict[selected_cell_type] if gene in adata.var_names]

    n_rows = len(all_genes_to_plot)
    n_cols = 3  # Always 3 plots per row

    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 10*n_rows), dpi=80)
        
        # Handle single row case
        if n_rows == 1:
            axes = [axes]
            
        # Create a temporary view with normalized data if available
        temp_adata = adata.copy()
        if 'for_cell_typist' in adata.layers:
            temp_adata.X = adata.layers['for_cell_typist']

        for i, gene in enumerate(all_genes_to_plot):
            row_axes = axes[i]  # Use integer index instead of gene name

            try:
                # Plot the marker gene using normalized data
                sc.pl.umap(temp_adata, color=gene, ax=row_axes[0], 
                           title=f"{gene} ({selected_cell_type})", cmap='Reds', show=False,
                           size=150)
                row_axes[0].set_xlabel("UMAP1", fontsize=20)
                row_axes[0].set_ylabel("UMAP2", fontsize=20)
                row_axes[0].tick_params(axis='both', which='major', labelsize=18)
                row_axes[0].set_title(f"{gene} ({selected_cell_type})", fontsize=26)

                # Plot leiden_0.38
                sc.pl.umap(adata, color='leiden_0.38', ax=row_axes[1], 
                            title='leiden_0.38', show=False, size=150, legend_loc = 'none')
                row_axes[1].set_xlabel("UMAP1", fontsize=20)
                row_axes[1].set_ylabel("UMAP2", fontsize=20)
                row_axes[1].tick_params(axis='both', which='major', labelsize=18)
                row_axes[1].set_title('leiden_0.38', fontsize=26)

                # Plot majority_voting
                sc.pl.umap(adata, color='majority_voting', ax=row_axes[2], 
                            title='majority_voting', show=False, size=150, legend_fontsize='xx-large', legend_loc = 'right margin')
                row_axes[2].set_xlabel("UMAP1", fontsize=20)
                row_axes[2].set_ylabel("UMAP2", fontsize=20)
                row_axes[2].tick_params(axis='both', which='major', labelsize=18)
                row_axes[2].set_title('majority_voting', fontsize=26)

            except Exception as e:
                print(f"Error plotting: {e}")

        plt.tight_layout()
        plt.show()