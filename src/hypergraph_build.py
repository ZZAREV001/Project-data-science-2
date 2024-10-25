import pandas as pd
import hypernetx as hnx
import json
import os
import numpy as np
import traceback

def create_numeric_edges(df, column, num_bins=5, method='quantile'):
    """
    Create edges based on numeric columns using binning
    """
    if method == 'quantile':
        labels = [f'{column}_q{i+1}' for i in range(num_bins)]
        bins = pd.qcut(df[column], q=num_bins, labels=labels, duplicates='drop')
    else:  # method == 'range'
        labels = [f'{column}_bin{i+1}' for i in range(num_bins)]
        bins = pd.cut(df[column], bins=num_bins, labels=labels)
    
    edges = {}
    for label in labels:
        recipes = df[bins == label]['recipe'].astype(str).tolist()
        if recipes:  # Only add non-empty edges
            edges[label] = recipes
    return edges

def create_serving_size_edges(df):
    """
    Create edges based on serving sizes
    """
    edges = {}
    serving_sizes = df['servings'].unique()
    for size in serving_sizes:
        recipes = df[df['servings'] == size]['recipe'].astype(str).tolist()
        edges[f'serving_size_{int(size)}'] = recipes
    return edges

def create_nutritional_edges(df, column, thresholds=None):
    """
    Create edges based on nutritional content with multiple thresholds
    """
    if thresholds is None:
        mean = df[column].mean()
        std = df[column].std()
        thresholds = {
            'very_low': mean - std,
            'low': mean - 0.5*std,
            'medium': mean,
            'high': mean + 0.5*std,
            'very_high': mean + std
        }
    
    edges = {}
    for level, threshold in thresholds.items():
        if level in ['very_low', 'low']:
            recipes = df[df[column] <= threshold]['recipe'].astype(str).tolist()
        else:
            recipes = df[df[column] > threshold]['recipe'].astype(str).tolist()
        edges[f'{column}_{level}'] = recipes
    return edges

try:
    print("Script started")

    # Load your cleaned dataset
    df = pd.read_csv('/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed/cleaned_data.csv')
    print("DataFrame loaded")
    print("DataFrame columns:", df.columns.tolist())

    # Initialize edges dictionary
    edges = {}

    # Create category edges
    for category in df['category'].unique():
        recipes = df[df['category'] == category]['recipe'].astype(str).tolist()
        edges[f'category_{category}'] = recipes
    print("Category edges created")

    # Create serving size edges
    serving_edges = create_serving_size_edges(df)
    edges.update(serving_edges)
    print("Serving size edges created")

    # Create nutritional edges
    nutritional_columns = ['calories', 'carbohydrate', 'sugar', 'protein']
    for col in nutritional_columns:
        nutritional_edges = create_nutritional_edges(df, col)
        edges.update(nutritional_edges)
    print("Nutritional edges created")

    # Create combination edges for interesting patterns
    # High protein and low carb
    high_protein_low_carb = df[
        (df['protein'] > df['protein'].mean()) & 
        (df['carbohydrate'] < df['carbohydrate'].mean())
    ]['recipe'].astype(str).tolist()
    edges['high_protein_low_carb'] = high_protein_low_carb

    # Create edges for recipe complexity based on number of nutrients
    df['nutrient_complexity'] = (
        (df['calories'] > df['calories'].mean()).astype(int) +
        (df['carbohydrate'] > df['carbohydrate'].mean()).astype(int) +
        (df['sugar'] > df['sugar'].mean()).astype(int) +
        (df['protein'] > df['protein'].mean()).astype(int)
    )
    for complexity in range(5):
        recipes = df[df['nutrient_complexity'] == complexity]['recipe'].astype(str).tolist()
        edges[f'complexity_{complexity}'] = recipes

    # Construct the hypergraph
    H = hnx.Hypergraph(edges)
    print("Hypergraph constructed")

    # Calculate some basic hypergraph metrics
    print(f"Number of nodes: {len(H.nodes)}")
    print(f"Number of edges: {len(H.edges)}")
    
    # Convert edges dictionary for JSON storage
    edges_json_compatible = {str(k): [str(v) for v in vs] for k, vs in edges.items()}

    # Add metadata about the hypergraph
    metadata = {
        "num_nodes": len(H.nodes),
        "num_edges": len(H.edges),
        "edge_types": list(edges_json_compatible.keys()),
        "creation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output_data = {
        "metadata": metadata,
        "edges": edges_json_compatible
    }

    # Define the output file path
    output_dir = '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed'
    output_file = os.path.join(output_dir, 'hypergraph_edges.json')

    # Save the edges dictionary to a JSON file
    print(f"Saving edges to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Hypergraph edges saved to {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()