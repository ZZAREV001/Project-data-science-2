import hypernetx as hnx
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import pandas as pd  
import numpy as np
import seaborn as sns

def visualize_recipe_hypergraph(json_file_path, csv_file_path, visualization_type='interactive'):
    """
    Visualize the recipe hypergraph with enhanced readability and interactivity.
    
    Parameters:
    -----------
    json_file_path : str
        Path to the JSON file containing hypergraph edges
    csv_file_path : str
        Path to the CSV file containing recipe details
    visualization_type : str
        'interactive' or 'static'
    """
    # Load the hypergraph data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Load recipe details
    df = pd.read_csv(csv_file_path)
    
    # Create category mapping for colors
    category_colors = {
        'Breakfast': '#FF9999',
        'Lunch/Snacks': '#66B2FF',
        'Dinner': '#99FF99',
        'Dessert': '#FFCC99',
        'Beverages': '#FF99CC',
        'Vegetable': '#99FFCC',
        'Meat': '#FF99FF',
        'Chicken': '#FFFF99',
        'Pork': '#99CCFF',
        'Potato': '#FFCC99',
        'One Dish Meal': '#CC99FF'
    }
    
    # Create more focused categories for visualization
    categories = {
        'recipe_types': [edge for edge in data['edges'].keys() if edge.startswith('cat_')][:5],
        'nutritional_high': [edge for edge in data['edges'].keys() if 'high' in edge][:3],
        'complexity': ['complexity_0', 'complexity_4']
    }
    
    if visualization_type == 'interactive':
        H = hnx.Hypergraph({k: data['edges'][k] for category in categories.values() for k in category})
        G = H.bipartite()
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge traces with distinct colors and better tooltips
        edge_traces = []
        for i, (category, edges) in enumerate(categories.items()):
            for edge_name in edges:
                if edge_name in data['edges']:
                    edge_x = []
                    edge_y = []
                    hover_text = []
                    
                    for node in data['edges'][edge_name]:
                        if (edge_name, node) in pos:
                            x0, y0 = pos[edge_name]
                            x1, y1 = pos[node]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                            
                            # Add recipe details to hover text
                            recipe_info = df[df['recipe'] == int(node)].iloc[0] if len(df[df['recipe'] == int(node)]) > 0 else None
                            if recipe_info is not None:
                                hover_text.append(
                                    f"Recipe: {node}<br>"
                                    f"Category: {recipe_info['category']}<br>"
                                    f"Calories: {recipe_info['calories']:.0f}<br>"
                                    f"Protein: {recipe_info['protein']:.1f}g<br>"
                                    f"Carbs: {recipe_info['carbohydrate']:.1f}g<br>"
                                    f"Sugar: {recipe_info['sugar']:.1f}g<br>"
                                    f"Servings: {recipe_info['servings']}"
                                )
                            else:
                                hover_text.append(f"Category: {edge_name}")
                    
                    edge_traces.append(
                        go.Scatter(
                            x=edge_x,
                            y=edge_y,
                            line=dict(width=2, color=category_colors.get(category, '#888')),
                            hoverinfo='text',
                            text=hover_text,
                            mode='lines',
                            name=edge_name,
                            showlegend=True
                        )
                    )
        
        # Create node trace with enhanced information
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        hover_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            
            # Different styling for category nodes vs recipe nodes
            if any(node.startswith(prefix) for prefix in ['cat_', 'complexity_', 'high_']):
                node_size.append(30)
                node_color.append('#FF0000')  # Red for category nodes
                hover_text.append(f"Category: {node}")
            else:
                recipe_info = df[df['recipe'] == int(node)].iloc[0] if len(df[df['recipe'] == int(node)]) > 0 else None
                if recipe_info is not None:
                    node_size.append(20)
                    node_color.append(category_colors.get(recipe_info['category'], '#888'))
                    hover_text.append(
                        f"Recipe: {node}<br>"
                        f"Category: {recipe_info['category']}<br>"
                        f"Calories: {recipe_info['calories']:.0f}<br>"
                        f"Protein: {recipe_info['protein']:.1f}g<br>"
                        f"Carbs: {recipe_info['carbohydrate']:.1f}g<br>"
                        f"Sugar: {recipe_info['sugar']:.1f}g<br>"
                        f"Servings: {recipe_info['servings']}"
                    )
                else:
                    node_size.append(15)
                    node_color.append('#888')
                    hover_text.append(f"Recipe: {node}")
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=node_text,
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='#888'),
                symbol='circle'
            ),
            textposition='top center',
            textfont=dict(size=8)
        )
        
        # Create figure with improved layout and legends
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title={
                    'text': 'Recipe Hypergraph Visualization<br>Hover over nodes for details',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=1000,
                width=1500,
                plot_bgcolor='rgb(248,248,248)',
                legend=dict(
                    title="Recipe Categories",
                    x=1.05,
                    y=0.5,
                    bordercolor='#888',
                    borderwidth=1
                ),
                annotations=[
                    dict(
                        text="Node Colors:<br>" + "<br>".join([f"{k}: {v}" for k, v in category_colors.items()]),
                        xref="paper",
                        yref="paper",
                        x=1.15,
                        y=0.8,
                        showarrow=False,
                        font=dict(size=8)
                    ),
                    dict(
                        text="Node Sizes:<br>Category: Large<br>Recipe: Medium<br>Connection: Small",
                        xref="paper",
                        yref="paper",
                        x=1.15,
                        y=0.2,
                        showarrow=False,
                        font=dict(size=8)
                    )
                ]
            )
        )
        
        fig.show()

    else:
        # Static visualization with matplotlib
        plt.figure(figsize=(20, 15))
        
        # Use distinct colors for categories
        colors = list(category_colors.values())
        
        # Create a subset of the hypergraph
        selected_edges = {k: data['edges'][k] 
                        for category in categories.values() 
                        for k in category 
                        if k in data['edges']}
        
        H = hnx.Hypergraph(selected_edges)
        
        # Draw with improved parameters
        hnx.draw(H,
                with_node_labels=True,
                with_edge_labels=True,
                node_size=200,
                edge_width=2,
                node_label_size=6,
                edge_label_size=8,
                with_node_counts=False,
                layout_kwargs={'seed': 42})
        
        # Add comprehensive legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, label=cat, markersize=10)
            for cat, color in category_colors.items()
        ]
        plt.legend(handles=legend_elements, 
                  title='Recipe Categories',
                  loc='center left', 
                  bbox_to_anchor=(1, 0.5))
        
        plt.title('Recipe Categories and Relationships\nNode size indicates connectivity', 
                 fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.show()

def analyze_cleaned_csv(csv_file_path):
    """
    Analyze the cleaned CSV dataset.
    """
    try:
        # Load the cleaned dataset
        df = pd.read_csv(csv_file_path)
        
        # Example analyses:
        # 1. Summary statistics
        print("Summary Statistics:")
        print(df.describe())
        
        # 2. Distribution plots (histograms)
        numeric_columns = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(numeric_columns):
            axes[idx].hist(df[col], bins=30)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # 3. Correlation matrix
        print("\nCorrelation Matrix:")
        correlation_matrix = df[numeric_columns].corr()
        print(correlation_matrix)
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        # 4. Category analysis
        print("\nCategory Distribution:")
        category_counts = df['category'].value_counts()
        print(category_counts)
        
        # Plot category distribution
        plt.figure(figsize=(12, 6))
        category_counts.plot(kind='bar')
        plt.title('Distribution of Recipes by Category')
        plt.xlabel('Category')
        plt.ylabel('Number of Recipes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # 5. Boxplots for numerical features by category
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, col in enumerate(['calories', 'carbohydrate', 'sugar', 'protein']):
            sns.boxplot(x='category', y=col, data=df, ax=axes[idx])
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
            axes[idx].set_title(f'{col} by Category')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in analyze_cleaned_csv: {e}")
        import traceback
        traceback.print_exc()
    
def visualize_selected_categories(file_path):
    """
    Function to visualize specific categories of a hypergraph in a simpler way.
    Categories visualized: high_traffic, high_sugar, and cat_Vegetable.
    """
    # Load hypergraph data from JSON file
    with open(file_path, 'r') as f:
        hypergraph_edges = json.load(f)
    print("Hypergraph edges loaded from JSON.")

    # Filter hypergraph edges based on selected categories
    selected_categories = ['high_traffic', 'high_sugar', 'cat_Vegetable']
    filtered_edges = {
        category: hypergraph_edges['edges'][category]
        for category in selected_categories if category in hypergraph_edges['edges']
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
    G = H_filtered.bipartite()

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    print("Matplotlib figure and axes created.")

    # Use a spring layout for better readability
    pos = nx.spring_layout(G)

    # Draw the graph using NetworkX
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_size=150,
        node_color='skyblue',
        edge_color='gray',
        font_size=8
    )
    print("Simplified hypergraph plotted using NetworkX.")

    # Set title and adjust layout for better readability
    plt.title("Simplified Visualization of Selected Categories", fontsize=14)
    plt.tight_layout(pad=2.0)
    plt.show()
    print("Plot displayed.")

def visualize_recipe_hypergraph(json_file_path, csv_file_path, visualization_type='interactive'):
    """
    Visualize the recipe hypergraph with enhanced readability and interactivity.
    """
    # Load the hypergraph data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Load recipe details
    df = pd.read_csv(csv_file_path)
    
    # Create category mapping for colors
    category_colors = {
        'Breakfast': '#FF9999',
        'Lunch/Snacks': '#66B2FF',
        'Dinner': '#99FF99',
        'Dessert': '#FFCC99',
        'Beverages': '#FF99CC',
        'Vegetable': '#99FFCC',
        'Meat': '#FF99FF',
        'Chicken': '#FFFF99',
        'Pork': '#99CCFF',
        'Potato': '#FFCC99',
        'One Dish Meal': '#CC99FF'
    }

    # Create more focused categories for visualization
    categories = {
        'recipe_types': [edge for edge in data['edges'].keys() if edge.startswith('category_')][:5],
        'nutritional_high': [edge for edge in data['edges'].keys() if edge.startswith(('calories_', 'carbohydrate_', 'sugar_', 'protein_'))][:3],
        'complexity': ['complexity_0', 'complexity_4']
    }

    if visualization_type == 'interactive':
        H = hnx.Hypergraph({k: data['edges'][k] for category in categories.values() for k in category})
        G = H.bipartite()
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for i, (category, edges) in enumerate(categories.items()):
            for edge_name in edges:
                if edge_name in data['edges']:
                    edge_x = []
                    edge_y = []
                    hover_text = []
                    
                    for node in data['edges'][edge_name]:
                        if (edge_name, node) in pos:
                            x0, y0 = pos[edge_name]
                            x1, y1 = pos[node]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                            
                            try:
                                recipe_id = int(node)
                                recipe_info = df[df['recipe'] == recipe_id].iloc[0] if len(df[df['recipe'] == recipe_id]) > 0 else None
                                if recipe_info is not None:
                                    hover_text.append(
                                        f"Recipe: {node}<br>"
                                        f"Category: {recipe_info['category']}<br>"
                                        f"Calories: {recipe_info['calories']:.0f}<br>"
                                        f"Protein: {recipe_info['protein']:.1f}g<br>"
                                        f"Carbs: {recipe_info['carbohydrate']:.1f}g<br>"
                                        f"Sugar: {recipe_info['sugar']:.1f}g<br>"
                                        f"Servings: {recipe_info['servings']}"
                                    )
                                else:
                                    hover_text.append(f"Recipe: {node}")
                            except ValueError:
                                hover_text.append(f"Category: {node}")
                    
                    edge_traces.append(
                        go.Scatter(
                            x=edge_x,
                            y=edge_y,
                            line=dict(width=2, color=category_colors.get(category, '#888')),
                            hoverinfo='text',
                            text=hover_text,
                            mode='lines',
                            name=edge_name,
                            showlegend=True
                        )
                    )
        
        # Create node trace
        node_x, node_y, node_text, node_size, node_color, hover_text = [], [], [], [], [], []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            
            if isinstance(node, str) and any(node.startswith(prefix) for prefix in 
                ['category_', 'complexity_', 'calories_', 'carbohydrate_', 'sugar_', 'protein_']):
                node_size.append(30)
                node_color.append('#FF0000')
                hover_text.append(f"Category: {node}")
            else:
                try:
                    recipe_id = int(node)
                    recipe_info = df[df['recipe'] == recipe_id].iloc[0] if len(df[df['recipe'] == recipe_id]) > 0 else None
                    if recipe_info is not None:
                        node_size.append(20)
                        node_color.append(category_colors.get(recipe_info['category'], '#888'))
                        hover_text.append(
                            f"Recipe: {node}<br>"
                            f"Category: {recipe_info['category']}<br>"
                            f"Calories: {recipe_info['calories']:.0f}<br>"
                            f"Protein: {recipe_info['protein']:.1f}g<br>"
                            f"Carbs: {recipe_info['carbohydrate']:.1f}g<br>"
                            f"Sugar: {recipe_info['sugar']:.1f}g<br>"
                            f"Servings: {recipe_info['servings']}"
                        )
                    else:
                        node_size.append(15)
                        node_color.append('#888')
                        hover_text.append(f"Recipe: {node}")
                except ValueError:
                    node_size.append(15)
                    node_color.append('#888')
                    hover_text.append(f"Node: {node}")
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=node_text,
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='#888'),
                symbol='circle'
            ),
            textposition='top center',
            textfont=dict(size=8)
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title={
                    'text': 'Recipe Hypergraph Visualization<br>Hover over nodes for details',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=1000,
                width=1500,
                plot_bgcolor='rgb(248,248,248)',
                legend=dict(
                    title="Recipe Categories",
                    x=1.05,
                    y=0.5,
                    bordercolor='#888',
                    borderwidth=1
                )
            )
        )
        
        fig.show()
    else:
        # Static visualization code here
        plt.figure(figsize=(20, 15))
        colors = list(category_colors.values())
        selected_edges = {k: data['edges'][k] 
                        for category in categories.values() 
                        for k in category 
                        if k in data['edges']}
        H = hnx.Hypergraph(selected_edges)
        hnx.draw(H,
                with_node_labels=True,
                with_edge_labels=True,
                node_size=200,
                edge_width=2,
                node_label_size=6,
                edge_label_size=8,
                with_node_counts=False,
                layout_kwargs={'seed': 42})
        plt.title('Recipe Categories and Relationships\nNode size indicates connectivity', 
                 fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()

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

def plot_servings_outliers_by_category(data, Q1, Q3):
    """
    Plots the distribution of servings outliers by category.
    
    Parameters:
    - data: pd.DataFrame, the dataset containing the recipes data.
    - Q1: pd.Series, first quartile values of the numerical columns.
    - Q3: pd.Series, third quartile values of the numerical columns.
    
    Returns:
    - None, shows a boxplot of servings outliers by category.
    """
    # Calculating IQR
    IQR = Q3 - Q1

    # Filtering outliers data for servings
    outliers_data = data[((data['servings'] < (Q1['servings'] - 1.5 * IQR['servings'])) |
                          (data['servings'] > (Q3['servings'] + 1.5 * IQR['servings'])))]
    
    # Plotting the distribution of servings outliers by category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='category', y='servings', data=outliers_data)
    plt.xticks(rotation=45)
    plt.title('Distribution of Servings Outliers by Category')
    plt.xlabel('Category')
    plt.ylabel('Servings')
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to execute analysis.
    """
    # Set matplotlib backend
    import matplotlib
    matplotlib.use('TkAgg')
    
    # For Plotly, ensure browser display
    import plotly.io as pio
    pio.renderers.default = 'browser'
    
    # Paths to your data files
    json_file_path = '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed/hypergraph_edges.json'  
    csv_file_path = '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed/cleaned_data.csv'        
    
    try:
        # Analyze cleaned CSV
        print("\nAnalyzing Cleaned CSV Data...")
        analyze_cleaned_csv(csv_file_path)
        
        # Visualize hypergraph categories
        print("\nVisualizing Hypergraph Categories...")
        visualize_selected_categories(json_file_path)

        # Generate improved visualization
        print("\nGenerating Enhanced Hypergraph Visualization...")
        visualize_recipe_hypergraph(json_file_path, csv_file_path, visualization_type='interactive')

        # Perform correlation analysis
        print("\nPerforming Correlation Analysis from hypergraph...")
        perform_correlation_analysis(json_file_path)

        # Perform correlation analysis from CSV
        print("\nPerforming Correlation Analysis from CSV...")
        df = pd.read_csv(csv_file_path)
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        plot_servings_outliers_by_category(df, Q1, Q3)
        
    except Exception as e:
        print(f"An error occurred in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()