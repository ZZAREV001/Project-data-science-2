import pandas as pd
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory
project_root = os.path.dirname(script_dir)

# Define data directories
raw_data_dir = os.path.join(project_root, 'data', 'raw')
processed_data_dir = os.path.join(project_root, 'data', 'processed')

# Ensure the processed data directory exists
os.makedirs(processed_data_dir, exist_ok=True)

# Define the path to the local CSV file
local_csv_path = os.path.join(raw_data_dir, 'recipe-site-traffic.csv')

# Load the CSV file with the correct delimiter and encoding
try:
    df = pd.read_csv(local_csv_path, sep=';', encoding='utf-8-sig')
    print("Data loaded successfully from local file")
    print("DataFrame columns:", df.columns.tolist())
except Exception as e:
    print(f"An error occurred: {e}")
    df = None  # Ensure df is defined


# Function Definitions:

def handle_missing_values(df):
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values per column before handling:\n", missing_values)
    
    # Impute missing numerical values with mean or median
    numerical_columns = ['calories', 'carbohydrate', 'sugar', 'protein']
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Handle missing values in 'high_traffic'
    # Option 1: Fill with the mode
    df['high_traffic'].fillna(df['high_traffic'].mode()[0], inplace=True)
    
    # Option 2: Fill with a new category 'Unknown' and update mapping in encoding function
    # df['high_traffic'].fillna('Unknown', inplace=True)
    
    # After handling missing values, check again
    missing_values_after = df.isnull().sum()
    print("Missing values per column after handling:\n", missing_values_after)
    
    return df


def remove_duplicates(df):
    # Check for duplicates
    duplicates = df.duplicated()
    print(f"Number of duplicate rows: {duplicates.sum()}")

    # Remove duplicate rows
    df_no_duplicates = df.drop_duplicates()

    return df_no_duplicates

def correct_data_types(df):
    # Ensure numerical columns are of numeric data types
    numeric_columns = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Convert 'recipe' to string if it's an identifier
    df['recipe'] = df['recipe'].astype(str)

    return df

def standardize_categories(df):
    # Strip whitespace and convert to title case for consistency
    df['category'] = df['category'].str.strip().str.title()

    # Define the expected categories
    expected_categories = [
        'Lunch/Snacks', 'Beverages', 'Potato', 'Vegetable', 'Meat',
        'Chicken', 'Pork', 'Dessert', 'Breakfast', 'One Dish Meal'
    ]

    # Replace any variations or typos
    df['category'] = df['category'].replace({
        # Add mappings if there are known typos
    })

    # Filter out unexpected categories
    df = df[df['category'].isin(expected_categories)]

    return df

def handle_outliers(df):
    # For each numeric column, remove outliers beyond 3 standard deviations
    numeric_columns = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings']
    for col in numeric_columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]

    return df

def validate_numerical_values(df):
    # Ensure no negative values in nutritional columns
    nutritional_columns = ['calories', 'carbohydrate', 'sugar', 'protein']
    for col in nutritional_columns:
        df = df[df[col] >= 0]

    # Ensure 'servings' is at least 1
    df = df[df['servings'] >= 1]

    return df

def encode_categorical_variables(df):
    # One-hot encode 'category'
    df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')
    
    # Encode 'high_traffic' as binary, handle 'Unknown' if added
    traffic_mapping = {'High': 1, 'Low': 0}
    # If 'Unknown' category exists, decide how to handle it, e.g., assign a new value or drop
    df_encoded['high_traffic'] = df_encoded['high_traffic'].map(traffic_mapping)
    
    # If 'high_traffic' still contains NaN (due to unmapped categories), handle it
    df_encoded['high_traffic'].fillna(-1, inplace=True)  # Assign -1 for 'Unknown' or missing
    
    # Ensure that one-hot encoded columns are of integer type
    category_columns = [col for col in df_encoded.columns if col.startswith('cat_')]
    df_encoded[category_columns] = df_encoded[category_columns].astype(int)
    
    return df_encoded

def preprocess_data(df):
    df = correct_data_types(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = standardize_categories(df)
    df = validate_numerical_values(df)
    df = handle_outliers(df)
    df = encode_categorical_variables(df)
    return df


# Main Execution Block

if df is not None:
    # Apply preprocessing
    df = preprocess_data(df)

    # Define the output file path
    output_file = os.path.join(processed_data_dir, 'cleaned_data.csv')

    # Save the cleaned DataFrame to CSV
    df.to_csv(output_file, index=False)

    print(f"Cleaned data saved to {output_file}")
else:
    print("Data loading failed. Exiting the script.")
