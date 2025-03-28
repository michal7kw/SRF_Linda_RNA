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
sc.pl.umap(adata, color = ['prob_conf_score', 'leiden_0.38', 'majority_voting'], legend_loc = 'right margin')

# %%
cell_types = gene_list.columns.tolist()
print(cell_types)

# %%
markers_dict = {col: gene_list[col].dropna().tolist() for col in gene_list.columns}
markers_dict

# %%
for selected_cell_type in markers_dict.keys():
    print(selected_cell_type)
    plot_marker_genes(adata, selected_cell_type, markers_dict)

# %%



