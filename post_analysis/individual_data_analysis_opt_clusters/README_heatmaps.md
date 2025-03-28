# Marker Gene Heatmap Generator

This script generates marker gene heatmaps for different clustering resolutions and cluster-specific biomarkers using saved `.h5ad` files.

## Usage

Basic usage:

```bash
python generate_marker_heatmaps.py --adata path/to/processed.h5ad --resolutions 0.1 0.3 0.5
```

### Arguments

- `--adata`: Path to the saved `.h5ad` file (required)
- `--resolutions`: List of resolution values to process (if not provided, will detect all available leiden clusterings)
- `--leiden_key`: Base name for leiden resolution keys (default: 'leiden')
- `--output_dir`: Directory to save heatmaps (default: 'heatmaps')
- `--n_genes`: Number of top marker genes per cluster (default: 5)
- `--cluster_specific`: Generate separate heatmaps for each cluster's markers (flag)
- `--show_figures`: Display generated figures (flag)
- `--marker_file`: Path to custom marker gene CSV file (optional)

### Examples

1. Generate heatmaps for all available resolutions:

```bash
python generate_marker_heatmaps.py --adata Emx1_Ctrl_processed.h5ad
```

2. Generate heatmaps for specific resolutions with more genes:

```bash
python generate_marker_heatmaps.py --adata Emx1_Ctrl_processed.h5ad --resolutions 0.3 0.5 --n_genes 10
```

3. Generate cluster-specific heatmaps:

```bash
python generate_marker_heatmaps.py --adata Emx1_Ctrl_processed.h5ad --resolutions 0.3 --cluster_specific
```

4. Use custom marker genes from a file:

```bash
python generate_marker_heatmaps.py --adata Emx1_Ctrl_processed.h5ad --marker_file my_markers.csv
```

## Custom Marker Files

You can provide a CSV file with custom marker genes in one of these formats:

1. Two columns: 'cluster' and 'gene'
2. Cluster IDs as column names, with genes listed in rows
3. Output format from scanpy with 'cluster' and 'names' columns

The script will automatically detect the format and load the marker genes accordingly. 