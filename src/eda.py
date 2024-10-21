import hypernetx as hnx
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import pandas as pd  
import numpy as np

def analyze_hypergraph(json_file_path):
    """
    Analyze and visualize the hypergraph from a JSON file.
    """
    # Load the hypergraph data from JSON file
    with open(json_file_path, 'r') as f:
        hypergraph_edges = json.load(f)
    
    # Filter hypergraph data to limit the number of hyperedges
    selected_hyperedges = ['cat_Breakfast', 'cat_Chicken', 'high_traffic']
    selected_recipes = ['1', '2', '3', '4', '5']  # Replace with actual recipe IDs as strings
    
    filtered_edges = {}
    for key in selected_hyperedges:
        filtered_edges[key] = [recipe for recipe in hypergraph_edges.get(key, []) if recipe in selected_recipes]
    
    # Remove empty hyperedges
    filtered_edges = {k: v for k, v in filtered_edges.items() if v}
    
    H_filtered = hnx.Hypergraph(filtered_edges)
    
    # Prepare node labels (e.g., mapping recipe IDs to names)
    recipe_id_to_name = {
        '1': 'Pancakes',
        '2': 'Omelette',
        '3': 'Chicken Salad',
        '4': 'Smoothie',
        '5': 'French Toast'
        # Add more mappings as needed
    }
    
    node_labels = {node: recipe_id_to_name.get(node, node) for node in H_filtered.nodes}
    
    # Define node colors based on attributes
    # Assuming you have high traffic recipes in 'high_traffic' hyperedge
    high_traffic_recipes = set(hypergraph_edges.get('high_traffic', []))
    node_colors = ['red' if node in high_traffic_recipes else 'blue' for node in H_filtered.nodes]
    
    # Convert hypergraph to NetworkX graph for visualization
    G = H_filtered.dual().bipartite()
    
    # Generate positions for nodes
    pos = nx.spring_layout(G)
    
    # Create edge traces for Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node traces for Plotly
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node_labels.get(node, node))
        if node in high_traffic_recipes:
            node_color.append('red')
        else:
            node_color.append('blue')
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        text=node_text,
        mode='markers+text',
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=2))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Hypergraph Visualization',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    # Display the plot
    fig.show()

def analyze_cleaned_csv(csv_file_path):
    """
    Analyze the cleaned CSV dataset.
    """
    # Load the cleaned dataset
    df = pd.read_csv(csv_file_path)
    
    # Example analyses:
    # 1. Summary statistics
    print("Summary Statistics:")
    print(df.describe())
    
    # 2. Distribution plots (histograms)
    import matplotlib.pyplot as plt
    numeric_columns = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings']
    df[numeric_columns].hist(bins=15, figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    
    # 3. Correlation matrix
    print("Correlation Matrix:")
    print(df[numeric_columns].corr())
    
    # 4. Scatter plot matrix
    pd.plotting.scatter_matrix(df[numeric_columns], figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    
    # 5. Category counts
    category_columns = [col for col in df.columns if col.startswith('cat_')]
    category_counts = df[category_columns].sum().sort_values(ascending=False)
    print("Category Counts:")
    print(category_counts)
    
    # Plot category counts
    category_counts.plot(kind='bar', figsize=(12, 6))
    plt.title('Number of Recipes per Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Recipes')
    plt.tight_layout()
    plt.show()
    
def visualize_selected_categories(file_path):
    """
    Function to visualize specific categories of a hypergraph in a simpler way.
    Categories visualized: high_traffic, high_sugar, and cat_Vegetable.

    Parameters:
    - file_path (str): Path to the JSON file containing hypergraph edges.

    Returns:
    - None: Displays the visualization.
    """
    # Load hypergraph data from JSON file
    with open(file_path, 'r') as f:
        hypergraph_edges = json.load(f)
    print("Hypergraph edges loaded from JSON.")

    # Filter hypergraph edges based on selected categories
    selected_categories = ['high_traffic', 'high_sugar', 'cat_Vegetable']
    filtered_edges = {
        category: hypergraph_edges[category]
        for category in selected_categories if category in hypergraph_edges
    }
    print(f"Filtered edges: {filtered_edges}")

    # Check if there are any edges after filtering
    if not filtered_edges:
        print("No edges found for the selected categories.")
        return

    # Create the hypergraph from filtered data
    H_filtered = hnx.Hypergraph(filtered_edges)
    print("Hypergraph created from filtered edges.")

    # Convert hypergraph to a NetworkX graph for easier visualization
    G = H_filtered.bipartite()  # Converts to a bipartite graph representation

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Moderate figure size for simplicity
    print("Matplotlib figure and axes created.")

    # Use a spring layout for better readability
    pos = nx.spring_layout(G)

    # Draw the graph using NetworkX
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_size=150,  # Smaller nodes for a cleaner view
        node_color='skyblue',  # Uniform color for nodes
        edge_color='gray',  # Uniform color for edges
        font_size=8
    )
    print("Simplified hypergraph plotted using NetworkX.")

    # Set title and adjust layout for better readability
    plt.title("Simplified Visualization of Hypergraph for Selected Categories: high_traffic, high_sugar, cat_Vegetable", fontsize=14)
    plt.tight_layout(pad=2.0)  # Add some padding for spacing
    plt.show()
    print("Plot displayed.")

def perform_correlation_analysis(file_path):
    """
    Function to perform exploratory data analysis to determine if a correlation exists between high_traffic and high_sugar recipes.

    Parameters:
    - file_path (str): Path to the JSON file containing hypergraph edges.

    Returns:
    - None: Prints correlation analysis results.
    """
    # Load hypergraph data from JSON file
    with open(file_path, 'r') as f:
        hypergraph_edges = json.load(f)
    print("Hypergraph edges loaded from JSON.")

    # Extract high_traffic and high_sugar recipes
    high_traffic_recipes = set(hypergraph_edges.get('high_traffic', []))
    high_sugar_recipes = set(hypergraph_edges.get('high_sugar', []))

    # Create a DataFrame for correlation analysis
    all_recipes = list(high_traffic_recipes.union(high_sugar_recipes))
    data = {
        'recipe_id': all_recipes,
        'is_high_traffic': [1 if recipe in high_traffic_recipes else 0 for recipe in all_recipes],
        'is_high_sugar': [1 if recipe in high_sugar_recipes else 0 for recipe in all_recipes]
    }
    df = pd.DataFrame(data)
    print("Data prepared for correlation analysis:")
    print(df.head())

    # Perform correlation analysis
    correlation_matrix = df[['is_high_traffic', 'is_high_sugar']].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Interpret the correlation
    correlation_value = correlation_matrix.loc['is_high_traffic', 'is_high_sugar']
    if correlation_value > 0.5:
        print(f"Strong positive correlation ({correlation_value}) between high sugar content and high traffic.")
    elif correlation_value > 0.2:
        print(f"Moderate positive correlation ({correlation_value}) between high sugar content and high traffic.")
    elif correlation_value > 0:
        print(f"Weak positive correlation ({correlation_value}) between high sugar content and high traffic.")
    else:
        print(f"No significant positive correlation ({correlation_value}) between high sugar content and high traffic.")

def perform_correlation_analysis_from_csv(csv_file_path):
    """
    Function to perform exploratory data analysis to determine if a correlation exists between high_traffic and high_sugar recipes from a cleaned CSV dataset.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the cleaned dataset.

    Returns:
    - None: Prints correlation analysis results.
    """
    # Load cleaned data from CSV file
    df = pd.read_csv(csv_file_path)
    print("Cleaned dataset loaded from CSV.")

    # Check if the necessary columns are present
    if 'high_traffic' not in df.columns or 'sugar' not in df.columns:
        print("The dataset does not contain the required columns: 'high_traffic' and 'sugar'.")
        return

    # Create a binary column for high sugar content
    sugar_threshold = df['sugar'].mean() + df['sugar'].std()  # Defining a threshold as mean + std deviation
    df['is_high_sugar'] = df['sugar'] > sugar_threshold
    df['is_high_sugar'] = df['is_high_sugar'].astype(int)  # Convert boolean to int (1 for high sugar, 0 otherwise)

    # Perform correlation analysis
    correlation_matrix = df[['high_traffic', 'is_high_sugar']].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Interpret the correlation
    correlation_value = correlation_matrix.loc['high_traffic', 'is_high_sugar']
    if correlation_value > 0.5:
        print(f"Strong positive correlation ({correlation_value}) between high sugar content and high traffic.")
    elif correlation_value > 0.2:
        print(f"Moderate positive correlation ({correlation_value}) between high sugar content and high traffic.")
    elif correlation_value > 0:
        print(f"Weak positive correlation ({correlation_value}) between high sugar content and high traffic.")
    else:
        print(f"No significant positive correlation ({correlation_value}) between high sugar content and high traffic.")


def main():
    """
    Main function to execute analysis.
    """
    # Paths to your data files
    json_file_path = '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed/hypergraph_edges.json'  
    csv_file_path = '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed/cleaned_data.csv'        
    
    # Analyze hypergraph
    print("Analyzing Hypergraph...")
    analyze_hypergraph(json_file_path)
    
    # Analyze cleaned CSV
    print("\nAnalyzing Cleaned CSV Data...")
    analyze_cleaned_csv(csv_file_path)
    
    # Visualize hypergraph categories
    print("\nVisualizing Hypergraph Categories...")
    visualize_selected_categories(json_file_path)

    # Perform correlation analysis
    print("\nPerforming Correlation Analysis from hypergraph...")
    perform_correlation_analysis(json_file_path)

    # Perform correlation analysis from CSV
    print("\nPerforming Correlation Analysis from CSV...")
    perform_correlation_analysis_from_csv(csv_file_path)

if __name__ == "__main__":
    main()
