#!/usr/bin/env python3
"""
This script takes a template notebook and creates parameterized versions for each sample.
It then executes each notebook and saves the results.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

def create_parameterized_notebook(template_path, output_path, sample_name):
    """
    Create a parameterized notebook by replacing the placeholder with the actual sample name.
    
    Args:
        template_path (str): Path to the template notebook
        output_path (str): Path to save the parameterized notebook
        sample_name (str): Sample name to use
    """
    # Read the template notebook
    with open(template_path, 'r') as f:
        notebook = json.load(f)
    
    # Find the cell with the placeholder and replace it
    for cell in notebook['cells']:
        if 'source' in cell and any('SAMPLE_PLACEHOLDER' in line for line in cell['source']):
            # Replace the placeholder with the actual sample name
            new_source = []
            for line in cell['source']:
                if 'SAMPLE_PLACEHOLDER' in line:
                    new_source.append(line.replace('SAMPLE_PLACEHOLDER', sample_name))
                else:
                    new_source.append(line)
            cell['source'] = new_source
    
    # Save the parameterized notebook
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Created parameterized notebook for {sample_name} at {output_path}")

def execute_notebook(notebook_path, timeout=-1):
    """
    Execute a notebook using jupyter nbconvert.
    
    Args:
        notebook_path (str): Path to the notebook to execute
        timeout (int): Timeout in seconds, -1 for no timeout
    
    Returns:
        bool: True if execution was successful, False otherwise
    """
    print(f"Executing notebook: {notebook_path}")
    
    try:
        # Execute the notebook
        result = subprocess.run([
            'jupyter', 'nbconvert', 
            '--to', 'notebook',
            '--execute',
            '--inplace',
            f'--ExecutePreprocessor.timeout={timeout}',
            '--ExecutePreprocessor.allow_errors=False',
            '--ClearOutputPreprocessor.enabled=False',
            notebook_path
        ], check=True, capture_output=True, text=True)
        
        print(f"Successfully executed {notebook_path}")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error executing {notebook_path}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run parameterized notebooks for multiple samples')
    parser.add_argument('--template', default='template_notebook.ipynb', help='Path to template notebook')
    parser.add_argument('--samples', nargs='+', default=['Emx1_Ctrl', 'Emx1_Mut', 'Nestin_Ctrl', 'Nestin_Mut'], 
                        help='List of sample names to process')
    parser.add_argument('--timeout', type=int, default=-1, help='Execution timeout in seconds, -1 for no timeout')
    parser.add_argument('--force', action='store_true', help='Force overwrite of existing notebooks')
    args = parser.parse_args()
    
    # Get the absolute path to the template
    template_path = os.path.abspath(args.template)
    
    # Check if template exists
    if not os.path.exists(template_path):
        print(f"Error: Template notebook not found at {template_path}")
        sys.exit(1)
    
    # Process each sample
    for sample_name in args.samples:
        # Define the output notebook path
        output_path = os.path.join(os.path.dirname(template_path), f"{sample_name}.ipynb")
        
        # Check if output notebook already exists
        if os.path.exists(output_path) and not args.force:
            print(f"Warning: Output notebook {output_path} already exists. Use --force to overwrite.")
            continue
        
        # Create the parameterized notebook
        create_parameterized_notebook(template_path, output_path, sample_name)
        
        # Execute the notebook
        execute_notebook(output_path, args.timeout)

if __name__ == "__main__":
    main()
