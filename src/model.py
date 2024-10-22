import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


def clustering_analysis(data_json):
    """
    Function to perform clustering analysis on recipe data using K-Means.
    
    Parameters:
    - data_json (str): Path to the JSON file containing hypergraph edges.
    
    Returns:
    - None: Displays plots and prints cluster analysis results.
    """
    # Load the JSON data
    with open(data_json, 'r') as f:
        data_dict = json.load(f)

    # Collect all recipe IDs
    all_recipe_ids = set()
    for recipes in data_dict.values():
        all_recipe_ids.update(recipes)

    # Initialize DataFrame
    df = pd.DataFrame({'recipe_id': list(all_recipe_ids)})
    df.set_index('recipe_id', inplace=True)

    # Initialize category columns with zeros
    for category in data_dict.keys():
        df[category] = 0

    # Fill in the category memberships
    for category, recipes in data_dict.items():
        df.loc[df.index.intersection(recipes), category] = 1

    # Exclude 'high_traffic' and 'high_sugar' from features
    features = df.drop(columns=['high_traffic', 'high_sugar'], errors='ignore')

    # Determine the optimal number of clusters using the Elbow Method
    wcss = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(8, 4))
    plt.plot(K, wcss, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    # Let's assume k=3 for this example
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(features)

    # Add cluster labels to the DataFrame
    df['cluster'] = clusters

    # Analyze cluster sizes
    print("Cluster sizes:")
    print(df['cluster'].value_counts())

    # Calculate cluster profiles
    cluster_profiles = df.groupby('cluster').mean()

    # Visualize cluster profiles
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_profiles.T, annot=True, cmap='viridis')
    plt.title('Cluster Profiles')
    plt.xlabel('Cluster')
    plt.ylabel('Category')
    plt.show()

    # Analyze 'high_sugar' distribution across clusters
    if 'high_sugar' in df.columns:
        cluster_high_sugar = df.groupby('cluster')['high_sugar'].mean()
        print("Average 'high_sugar' per cluster:")
        print(cluster_high_sugar)


def clustering_analysis_csv(csv_file_path):
    """
    Function to perform clustering analysis on recipe data using K-Means.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing recipe data.

    Returns:
    - None: Displays plots and prints cluster analysis results.
    """
    # Load the CSV data
    df = pd.read_csv(csv_file_path)

    # Handle missing values if any
    df.dropna(inplace=True)

    # Convert numerical columns to numeric types
    numerical_cols = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings']
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

    # Convert category columns to integers
    category_cols = ['cat_Beverages', 'cat_Breakfast', 'cat_Chicken', 'cat_Dessert',
                     'cat_Lunch/Snacks', 'cat_Meat', 'cat_One Dish Meal', 'cat_Pork',
                     'cat_Potato', 'cat_Vegetable']
    df[category_cols] = df[category_cols].astype(int)

    # Feature scaling for numerical columns
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_cols])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_cols, index=df.index)

    # Combine scaled numerical features with categorical features
    features = pd.concat([scaled_numerical_df, df[category_cols]], axis=1)

    # Determine the optimal number of clusters using the Elbow Method
    wcss = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(8, 4))
    plt.plot(K, wcss, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    # Let's assume k=3 for this example
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(features)

    # Add cluster labels to the DataFrame
    df['cluster'] = clusters

    # Analyze cluster sizes
    print("Cluster sizes:")
    print(df['cluster'].value_counts())

    # Calculate cluster profiles
    cluster_profiles = df.groupby('cluster').mean()

    # Visualize cluster profiles
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_profiles.T, annot=True, cmap='viridis')
    plt.title('Cluster Profiles')
    plt.xlabel('Cluster')
    plt.ylabel('Features')
    plt.show()

    # If you have a target variable like 'high_traffic', analyze their distribution
    if 'high_traffic' in df.columns:
        cluster_high_traffic = df.groupby('cluster')['high_traffic'].mean()
        print("Average 'high_traffic' per cluster:")
        print(cluster_high_traffic)


def logistic_regression_analysis(csv_file_path):
    """
    Function to perform logistic regression analysis on recipe data.
    """
    # Load the CSV data
    df = pd.read_csv(csv_file_path)

    # Data preprocessing
    df.dropna(inplace=True)
    numerical_cols = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings']
    category_cols = ['cat_Beverages', 'cat_Breakfast', 'cat_Chicken', 'cat_Dessert',
                     'cat_Lunch/Snacks', 'cat_Meat', 'cat_One Dish Meal', 'cat_Pork',
                     'cat_Potato', 'cat_Vegetable']
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    df[category_cols] = df[category_cols].astype(int)
    df['high_traffic'] = df['high_traffic'].astype(int)

    # Check the distribution of 'high_traffic'
    print("Value counts of 'high_traffic':")
    print(df['high_traffic'].value_counts())

    # Ensure there are at least two classes
    if df['high_traffic'].nunique() < 2:
        print("The target variable 'high_traffic' needs to have at least two classes.")
        print("Please ensure that your dataset includes recipes with 'high_traffic' equal to 0 and 1.")
        return

    # Feature scaling
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_cols])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_cols, index=df.index)

    # Combine features
    X = pd.concat([scaled_numerical_df, df[category_cols]], axis=1)
    y = df['high_traffic']

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def main():
    """
    Main function to prompt user to select an analysis algorithm.
    """
    print("Select the algorithm to perform analysis:")
    print("1. Clustering Analysis for the hypergraph (K-Means)")
    print("2. Clustering Analysis for the CSV data (K-Means)")
    print("3. Logistic Regression Analysis for the CSV data")
    choice = input("Enter the number of your choice: ")

    if choice == '1':
        data_json = "/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed/hypergraph_edges.json"
        clustering_analysis(data_json)
    elif choice == '2':
        csv_file_path = '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed/cleaned_data.csv'
        clustering_analysis_csv(csv_file_path)
    elif choice == '3':
        csv_file_path = '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI-ML-courses/Projets/Project-data-science-2/data/processed/cleaned_data.csv'
        logistic_regression_analysis(csv_file_path)
    else:
        print("Invalid choice. Please select a valid option.")   


if __name__ == "__main__":
    main()
