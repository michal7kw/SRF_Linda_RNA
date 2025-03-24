import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import anndata as ad
import warnings
from celltypist import models, annotate

# Set up directory for results
results_dir = "annotation_results"
os.makedirs(results_dir, exist_ok=True)

# Specific model path - using local model
MOUSE_BRAIN_MODEL = "Mouse_Isocortex_Hippocampus.pkl"

def prepare_data_for_celltypist(adata):
    """
    Prepare data for CellTypist annotation.
    CellTypist requires log1p normalized data with 10,000 counts per cell.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with raw or normalized data
        
    Returns:
    --------
    adata_norm : AnnData
        AnnData object prepared for CellTypist
    """
    print("Preparing data for CellTypist annotation...")
    
    # Make a copy to avoid changing the original
    adata_norm = adata.copy()
    
    # Check if the data has 'counts' layer
    if 'counts' in adata.layers:
        print("Using raw counts from 'counts' layer...")
        adata_norm.X = adata.layers['counts'].copy()
    
    # Normalize to 10,000 counts per cell as required by CellTypist
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    
    # Quick check that normalization worked correctly
    counts_after_norm = np.expm1(adata_norm.X).sum(axis=1)
    
    # Basic QC check
    if np.mean(counts_after_norm) < 9000 or np.mean(counts_after_norm) > 11000:
        warnings.warn("Normalization may not have worked as expected. Check your data.")
        
    return adata_norm

def explore_model(model_path):
    """
    Load the CellTypist model and explore its properties and cell types.
    
    Parameters:
    -----------
    model_path : str
        Path to the CellTypist model
        
    Returns:
    --------
    model : CellTypist Model
        Loaded CellTypist model
    """
    print(f"Loading CellTypist model: {model_path}")
    
    try:
        # Load the model
        model = models.Model.load(model_path)
        
        # Print model info
        print(f"Model: {os.path.basename(model_path)}")
        print(f"Number of cell types: {len(model.cell_types)}")
        print(f"Number of genes: {model.n_genes}")
        
        print("\nTop cell types:")
        for i, cell_type in enumerate(model.cell_types[:20]):
            print(f"  {i+1}. {cell_type}")
        
        if len(model.cell_types) > 20:
            print(f"  ... and {len(model.cell_types) - 20} more")
        
        # Extract some key marker genes
        print("\nExtracting markers for key cell types...")
        for cell_type in model.cell_types:
            if any(region in cell_type.lower() for region in ['hippo', 'ca1', 'dentate', 'cortex', 'layer']):
                try:
                    markers = model.extract_top_markers(cell_type, 5)
                    print(f"\nTop 5 markers for {cell_type}:")
                    for marker in markers:
                        print(f"  - {marker}")
                except:
                    print(f"Could not extract markers for {cell_type}")
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_mouse_brain_markers():
    """
    Get marker genes for mouse hippocampus and cortex cell types.
    
    Returns:
    --------
    markers : dict
        Dictionary of cell type markers for visualization
    """
    # These markers are based on Allen Brain Atlas and literature
    markers = {
        # Excitatory neurons
        "Excitatory_neurons": ["Slc17a7", "Neurod6", "Nrgn", "Camk2a", "Satb2"],
        
        # Inhibitory neurons
        "Inhibitory_neurons": ["Gad1", "Gad2", "Slc32a1", "Pvalb", "Sst", "Vip"],
        
        # Hippocampal cells
        "Hippocampal_CA1": ["Wfs1", "Nr3c2", "Spink8"],
        "Hippocampal_CA2": ["Amigo2", "Pcp4", "Cacng5", "Rgs14"],
        "Hippocampal_CA3": ["Bok", "Kcnq5", "Cpne4", "Grik4", "Prss12"],
        "Dentate_gyrus": ["Prox1", "Lrrtm4", "Dsp", "Ctgf", "Drd1a"],
        
        # Cortical layers
        "Layer_2_3": ["Cux1", "Cux2", "Lamp5", "Ptgs2"],
        "Layer_4": ["Rorb", "Rspo1", "Scnn1a"],
        "Layer_5": ["Bcl6", "Fezf2", "Deptor", "Ctip2", "Hsd11b1"],
        "Layer_6": ["Foxp2", "Ctgf", "Nxph2", "Tle4"],
        
        # Glial cells
        "Astrocytes": ["Aqp4", "Gfap", "Aldh1l1", "Slc1a2", "Slc1a3"],
        "Oligodendrocytes": ["Mbp", "Mog", "Plp1", "Olig1", "Sox10"],
        "OPCs": ["Pdgfra", "Cspg4", "Sox10"],
        "Microglia": ["Csf1r", "Cx3cr1", "P2ry12", "Tmem119", "Hexb"],
    }
    
    return markers

def cell_annotation_with_celltypist(adata, model_path, majority_voting=True, prob_threshold=0.5):
    """
    Perform cell type annotation using CellTypist.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with data
    model_path : str
        Path to the CellTypist model
    majority_voting : bool, optional
        Whether to use majority voting
    prob_threshold : float, optional
        Probability threshold for multi-label classification
        
    Returns:
    --------
    adata : AnnData
        AnnData with cell type annotations
    predictions : CellTypist AnnotationResult
        Raw prediction results
    """
    print(f"Starting cell type annotation with model: {model_path}")
    
    # Prepare data for CellTypist
    adata_norm = prepare_data_for_celltypist(adata)
    
    # Run CellTypist annotation
    print(f"Running CellTypist with majority_voting={majority_voting}, prob_threshold={prob_threshold}")
    predictions = annotate(
        adata_norm, 
        model=model_path,
        majority_voting=majority_voting,
        mode='prob match',  # Use probability-based matching for multi-label classification
        p_thres=prob_threshold
    )
    
    # Add annotations to original adata
    predictions.to_adata(adata)
    
    # Also add probability scores for key cell types
    predictions.to_adata(adata, insert_prob=True, prefix='prob_')
    
    print(f"CellTypist annotation completed")
    print(f"Identified cell types: {adata.obs['majority_voting'].value_counts().head(10).to_dict()}")
    
    return adata, predictions

def visualize_annotation_results(adata, output_dir=None):
    """
    Visualize CellTypist annotation results.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with CellTypist annotations
    output_dir : str, optional
        Directory to save plots
    """
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have dimensionality reduction
    if 'X_umap' not in adata.obsm:
        try:
            # Calculate neighborhood graph if not present
            if 'neighbors' not in adata.uns:
                sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        except Exception as e:
            print(f"Could not calculate UMAP: {e}")
            if 'X_pca' not in adata.obsm:
                sc.pp.pca(adata)
    
    # Plot cell type annotations
    try:
        # Cell type annotation plot
        if 'majority_voting' in adata.obs.columns:
            fig, ax = plt.subplots(figsize=(12, 10))
            sc.pl.umap(adata, color='majority_voting', ax=ax, legend_loc='on data', 
                      title="Cell Type Annotation (Majority Voting)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "celltypist_majority_voting.png"), dpi=300)
            plt.close()
        
        if 'predicted_labels' in adata.obs.columns:
            fig, ax = plt.subplots(figsize=(12, 10))
            sc.pl.umap(adata, color='predicted_labels', ax=ax, legend_loc='on data', 
                      title="Cell Type Annotation (Raw Prediction)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "celltypist_raw_prediction.png"), dpi=300)
            plt.close()
        
        # Confidence score plot
        if 'conf_score' in adata.obs.columns:
            fig, ax = plt.subplots(figsize=(12, 10))
            sc.pl.umap(adata, color='conf_score', ax=ax, 
                      title="Annotation Confidence Score", cmap='viridis')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "celltypist_confidence.png"), dpi=300)
            plt.close()
        
        # Get probability columns
        prob_columns = [col for col in adata.obs.columns if col.startswith('prob_')]
        
        # Find hippocampal and cortical cell types
        hippo_cols = [col for col in prob_columns if any(term in col.lower() for term in ['hippo', 'ca1', 'ca2', 'ca3', 'dentate'])]
        cortex_cols = [col for col in prob_columns if any(term in col.lower() for term in ['cortex', 'cortical', 'layer'])]
        
        # Plot top probabilities for hippocampus and cortex
        for region, cols, title in [
            ('hippocampus', hippo_cols, 'Hippocampal Cell Types'), 
            ('cortex', cortex_cols, 'Cortical Cell Types')
        ]:
            if cols:
                region_dir = os.path.join(output_dir, region)
                os.makedirs(region_dir, exist_ok=True)
                
                # Create a combined plot for top cell types
                top_cols = cols[:min(6, len(cols))]
                if len(top_cols) > 0:
                    try:
                        # Plot each cell type
                        for col in top_cols:
                            cell_type = col.replace('prob_', '')
                            fig, ax = plt.subplots(figsize=(12, 10))
                            sc.pl.umap(adata, color=col, ax=ax, title=f"{cell_type} Probability", 
                                      cmap='viridis', vmin=0, vmax=1)
                            plt.tight_layout()
                            plt.savefig(os.path.join(region_dir, f"{cell_type}_probability.png"), dpi=300)
                            plt.close()
                    except Exception as e:
                        print(f"Error plotting probabilities for {region}: {e}")
        
        # Plot marker genes
        markers = get_mouse_brain_markers()
        if markers:
            marker_dir = os.path.join(output_dir, "markers")
            os.makedirs(marker_dir, exist_ok=True)
            
            # Plot markers for hippocampus and cortex
            for region, cell_types in [
                ("Hippocampus", ["Hippocampal_CA1", "Hippocampal_CA3", "Dentate_gyrus"]),
                ("Cortex", ["Layer_2_3", "Layer_4", "Layer_5", "Layer_6"])
            ]:
                region_marker_dir = os.path.join(marker_dir, region.lower())
                os.makedirs(region_marker_dir, exist_ok=True)
                
                for cell_type in cell_types:
                    if cell_type in markers:
                        # Find genes that exist in the dataset
                        valid_genes = [gene for gene in markers[cell_type] if gene in adata.var_names]
                        
                        if valid_genes:
                            # Plot each marker gene
                            for gene in valid_genes[:3]:  # Plot top 3 marker genes
                                try:
                                    fig, ax = plt.subplots(figsize=(12, 10))
                                    sc.pl.umap(adata, color=gene, ax=ax, 
                                             title=f"{gene} ({cell_type})", cmap='viridis')
                                    plt.tight_layout()
                                    plt.savefig(os.path.join(region_marker_dir, f"{gene}_{cell_type}.png"), dpi=300)
                                    plt.close()
                                except Exception as e:
                                    print(f"Error plotting marker gene {gene}: {e}")
        
    except Exception as e:
        print(f"Error in visualization: {e}")

def extract_brain_regions(adata):
    """
    Extract cells annotated as hippocampus or cortex into separate AnnData objects.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with CellTypist annotations
        
    Returns:
    --------
    adata_regions : dict
        Dictionary with AnnData objects for each brain region
    """
    if 'majority_voting' not in adata.obs.columns:
        print("Annotation column 'majority_voting' not found")
        return {}
    
    # Create masks for different brain regions
    hippo_mask = adata.obs['majority_voting'].astype(str).str.contains('|'.join(['hippo', 'ca1', 'ca2', 'ca3', 'dentate']), case=False)
    cortex_mask = adata.obs['majority_voting'].astype(str).str.contains('|'.join(['cortex', 'cortical', 'layer']), case=False)
    
    # Create AnnData objects for each region
    adata_regions = {}
    
    if hippo_mask.sum() > 0:
        print(f"Extracting {hippo_mask.sum()} hippocampal cells")
        adata_regions['hippocampus'] = adata[hippo_mask].copy()
        
        # Calculate UMAP if needed
        if 'X_umap' not in adata_regions['hippocampus'].obsm:
            try:
                sc.pp.neighbors(adata_regions['hippocampus'])
                sc.tl.umap(adata_regions['hippocampus'])
            except:
                print("Could not calculate UMAP for hippocampal cells")
    
    if cortex_mask.sum() > 0:
        print(f"Extracting {cortex_mask.sum()} cortical cells")
        adata_regions['cortex'] = adata[cortex_mask].copy()
        
        # Calculate UMAP if needed
        if 'X_umap' not in adata_regions['cortex'].obsm:
            try:
                sc.pp.neighbors(adata_regions['cortex'])
                sc.tl.umap(adata_regions['cortex'])
            except:
                print("Could not calculate UMAP for cortical cells")
    
    return adata_regions

def analyze_region_subtypes(adata_regions):
    """
    Analyze and visualize subtypes within each brain region.
    
    Parameters:
    -----------
    adata_regions : dict
        Dictionary with AnnData objects for each brain region
    """
    for region, adata_region in adata_regions.items():
        print(f"Analyzing subtypes in {region}...")
        
        # Create output directory
        region_dir = os.path.join(results_dir, f"{region}_subtypes")
        os.makedirs(region_dir, exist_ok=True)
        
        # Plot UMAP with cell type annotations
        if 'majority_voting' in adata_region.obs.columns:
            fig, ax = plt.subplots(figsize=(12, 10))
            sc.pl.umap(adata_region, color='majority_voting', ax=ax, legend_loc='on data', 
                      title=f"{region.capitalize()} Cell Types")
            plt.tight_layout()
            plt.savefig(os.path.join(region_dir, f"{region}_subtypes.png"), dpi=300)
            plt.close()
        
        # Count subtypes
        if 'majority_voting' in adata_region.obs.columns:
            subtype_counts = adata_region.obs['majority_voting'].value_counts()
            
            # Create a bar plot of subtypes
            fig, ax = plt.subplots(figsize=(14, 8))
            subtype_counts.plot(kind='bar', ax=ax)
            ax.set_title(f"{region.capitalize()} Cell Type Counts")
            ax.set_ylabel("Number of cells")
            ax.set_xlabel("Cell Type")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(region_dir, f"{region}_subtype_counts.png"), dpi=300)
            plt.close()
            
            # Save counts to CSV
            subtype_counts.to_csv(os.path.join(region_dir, f"{region}_subtype_counts.csv"))
            
            print(f"Top cell types in {region}:")
            for subtype, count in subtype_counts.head(10).items():
                print(f"  {subtype}: {count} cells")

def main(adata_path=None):
    """
    Main function to run cell type annotation on a mouse brain scRNA-seq dataset
    using the Mouse_Isocortex_Hippocampus model.
    
    Parameters:
    -----------
    adata_path : str, optional
        Path to AnnData object file (.h5ad)
        
    Returns:
    --------
    adata : AnnData
        Annotated AnnData object
    """
    # Load data
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
            return None
    
    print(f"AnnData object contains {adata.n_obs} cells and {adata.n_vars} genes")
    
    # Verify model exists
    if not os.path.exists(MOUSE_BRAIN_MODEL):
        print(f"Model file {MOUSE_BRAIN_MODEL} not found in the current directory.")
        return adata
    
    # Explore model
    model = explore_model(MOUSE_BRAIN_MODEL)
    if model is None:
        print("Failed to load model. Please check the model file.")
        return adata
    
    # Run annotation
    adata, predictions = cell_annotation_with_celltypist(
        adata, 
        MOUSE_BRAIN_MODEL,
        majority_voting=True,
        prob_threshold=0.5
    )
    
    # Visualize results
    visualize_annotation_results(adata)
    
    # Extract brain regions
    adata_regions = extract_brain_regions(adata)
    
    # Analyze region subtypes
    analyze_region_subtypes(adata_regions)
    
    # Save annotated data
    output_path = os.path.join(results_dir, "annotated_mouse_brain.h5ad")
    adata.write(output_path)
    print(f"Annotated data saved to {output_path}")
    
    # Save region-specific data
    for region, adata_region in adata_regions.items():
        output_path = os.path.join(results_dir, f"{region}_cells.h5ad")
        adata_region.write(output_path)
        print(f"{region.capitalize()} cells saved to {output_path}")
    
    return adata

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Annotate mouse brain scRNA-seq data with CellTypist.')
    parser.add_argument('--adata', type=str, help='Path to AnnData file')
    args = parser.parse_args()
    
    # Run main function
    main(args.adata)