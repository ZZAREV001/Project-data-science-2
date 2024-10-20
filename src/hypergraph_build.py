print("Top of script")
import pandas as pd
import hypernetx as hnx
import json
import os
import traceback

try:
    print("Script started")

    # Load your cleaned dataset
    df = pd.read_csv('/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed/cleaned_data.csv')
    print("DataFrame loaded")
    print("DataFrame columns:", df.columns.tolist())

    # Initialize a dictionary to hold hyperedges
    edges = {}

    # Identify category columns (columns starting with 'cat_')
    category_columns = [col for col in df.columns if col.startswith('cat_')]
    print("Category columns found:", category_columns)

    # Create hyperedges for categories
    for category_col in category_columns:
        category_name = category_col.replace('cat_', '')
        # Get recipes belonging to this category
        recipes_in_category = df[df[category_col] == 1]['recipe'].astype(str).tolist()
        edges[f'cat_{category_name}'] = recipes_in_category
    print("Category hyperedges created")

    # Check if 'high_traffic' column exists
    if 'high_traffic' in df.columns:
        # Create hyperedges for high traffic recipes
        high_traffic_recipes = df[df['high_traffic'] == 1]['recipe'].astype(str).tolist()
        edges['high_traffic'] = high_traffic_recipes
        print("High traffic hyperedges created")
    else:
        print("Column 'high_traffic' not found in DataFrame. Skipping high traffic hyperedges.")

    # Check if 'sugar' column exists
    if 'sugar' in df.columns:
        # Create hyperedges for high sugar content
        sugar_threshold = df['sugar'].mean() + df['sugar'].std()
        high_sugar_recipes = df[df['sugar'] > sugar_threshold]['recipe'].astype(str).tolist()
        edges['high_sugar'] = high_sugar_recipes
        print("High sugar hyperedges created")
    else:
        print("Column 'sugar' not found in DataFrame. Skipping high sugar hyperedges.")

    # Construct the hypergraph
    H = hnx.Hypergraph(edges)
    print("Hypergraph constructed")

    # Convert the edges dictionary keys and values to strings
    edges_json_compatible = {str(k): [str(v) for v in vs] for k, vs in edges.items()}

    # Define the output file path
    output_dir = '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed'
    output_file = os.path.join(output_dir, 'hypergraph_edges.json')

    # Save the edges dictionary to a JSON file
    print(f"Saving edges to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(edges_json_compatible, f)
    print(f"Hypergraph edges saved to {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
