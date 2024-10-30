# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Set random seed for reproducibility
np.random.seed(42)

# Rest of your functions remain the same...

def prepare_data_regression(csv_file_path):
    """
    Prepare data for regression model to predict traffic levels
    """
    df = pd.read_csv(csv_file_path)
    
    # Create numerical features
    numerical_features = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings']
    
    # Create traffic score (composite metric)
    df['traffic_score'] = (
        df['calories'].rank(pct=True) * 0.3 +  # Higher calories often indicate more complex/interesting recipes
        df['protein'].rank(pct=True) * 0.3 +   # Protein content is often searched for
        df['servings'].rank(pct=True) * 0.2 +  # Serving size affects popularity
        pd.get_dummies(df['category']).apply(lambda x: x.rank(pct=True), axis=0).mean(axis=1) * 0.2  # Category popularity
    )
    
    # Create feature matrix
    X = pd.concat([
        df[numerical_features],
        pd.get_dummies(df['category'])
    ], axis=1)
    
    y = df['traffic_score']
    
    return X, y

def create_advanced_features(df):
    """
    Create more sophisticated features for prediction
    """
    # Nutritional density
    df['calorie_density'] = df['calories'] / df['servings']
    df['protein_density'] = df['protein'] / df['servings']
    
    # Macro ratios
    df['protein_carb_ratio'] = df['protein'] / (df['carbohydrate'] + 1)
    df['sugar_carb_ratio'] = df['sugar'] / (df['carbohydrate'] + 1)
    
    # Complexity indicators
    df['ingredient_complexity'] = (
        df['calories'].rank(pct=True) * 0.4 +
        df['protein'].rank(pct=True) * 0.3 +
        df['carbohydrate'].rank(pct=True) * 0.3
    )
    
    # Category popularity
    category_counts = df['category'].value_counts()
    df['category_popularity'] = df['category'].map(category_counts)
    
    return df

def analyze_data(df):
    """
    Analyze the dataset before modeling
    """
    print("\nData Analysis:")
    print(f"Total samples: {len(df)}")
    print("\nFeature Statistics:")
    print(df[['calories', 'carbohydrate', 'sugar', 'protein', 'servings']].describe())
    
    print("\nCategory Distribution:")
    print(df['category'].value_counts())
    
    # Visualize distributions
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(['calories', 'carbohydrate', 'sugar', 'protein']):
        plt.subplot(1, 4, i+1)
        sns.histplot(df[col], bins=30)
        plt.title(col)
    plt.tight_layout()
    plt.show()

def cluster_recipes(df, n_clusters=5):
    """
    Cluster recipes based on nutritional content
    """
    # Prepare features for clustering
    features = ['calories', 'carbohydrate', 'sugar', 'protein']
    X = df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['recipe_cluster'] = kmeans.fit_predict(X_scaled)
    
    return df

def analyze_clusters(df):
    """
    Analyze recipe clusters
    """
    cluster_stats = df.groupby(['recipe_cluster', 'category']).agg({
        'calories': 'mean',
        'protein': 'mean',
        'carbohydrate': 'mean',
        'sugar': 'mean',
        'servings': 'mean'
    }).round(2)
    
    return cluster_stats

def evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate multiple ML models
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_scores_mean': cv_scores.mean(),
                'cv_scores_std': cv_scores.std(),
                'classification_report': report,
                'confusion_matrix': conf_matrix
            }
            
            # Print feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                print(f"\nFeature Importance for {name}:")
                importances = model.feature_importances_
                for feat, imp in zip(feature_names, importances):
                    print(f"{feat}: {imp:.4f}")
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue
    
    return results

def main():
    """
    Main function to execute analysis pipeline
    """
    try:
        csv_file_path = 'data/processed/cleaned_data.csv'
        
        # Load data
        print("Loading data...")
        df = pd.read_csv(csv_file_path)
        
        # 1. Initial Data Analysis
        print("\nPerforming initial data analysis...")
        analyze_data(df)
        
        # 2. Feature Engineering
        print("\nCreating advanced features...")
        df_engineered = create_advanced_features(df)
        
        # 3. Clustering Analysis
        print("\nPerforming clustering analysis...")
        df_clustered = cluster_recipes(df_engineered)
        cluster_insights = analyze_clusters(df_clustered)
        print("\nCluster Analysis Results:")
        print(cluster_insights)
        
        # 4. Regression Modeling
        print("\nPreparing regression model...")
        X, y = prepare_data_regression(csv_file_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train regression models
        regression_models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42
            )
        }
        
        print("\nTraining regression models...")
        regression_results = {}
        for name, model in regression_models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate R² score
            score = r2_score(y_test, y_pred)
            regression_results[name] = score
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                print(f"\nFeature Importance for {name}:")
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                })
                print(feature_importance.sort_values('importance', ascending=False).head(10))
        
        print("\nRegression Model Performance (R² scores):")
        for name, score in regression_results.items():
            print(f"{name}: {score:.3f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()