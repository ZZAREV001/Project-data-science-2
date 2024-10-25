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
local_csv_path = os.path.join(raw_data_dir, 'recipe_site_traffic_2212.csv')

# Load the CSV file with the correct delimiter and encoding
try:
    df = pd.read_csv(local_csv_path)
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
    
    # Impute missing numerical values with median
    numerical_columns = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings']
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())
    
    # Handle missing values in 'high_traffic'
    df['high_traffic'] = df['high_traffic'].fillna(df['high_traffic'].mode()[0])
    
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
    # Add logging
    print("Original servings values:", df['servings'].unique())
    
    # Clean the servings column
    df['servings'] = df['servings'].astype(str).str.extract('(\d+)').astype(float)
    
    print("Cleaned servings values:", df['servings'].unique())
    
    # Rest of the function remains the same
    numeric_columns = ['calories', 'carbohydrate', 'sugar', 'protein']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df['recipe'] = pd.to_numeric(df['recipe'], errors='coerce')
    
    return df

def validate_categories(df):
    valid_categories = [
        'Lunch/Snacks', 'Beverages', 'Potato',
        'Vegetable', 'Meat', 'Chicken', 'Pork',
        'Dessert', 'Breakfast', 'One Dish Meal'
    ]
    df = df[df['category'].isin(valid_categories)]
    return df

# Add this function:
def validate_high_traffic(df):
    df['high_traffic'] = df['high_traffic'].astype(str)
    df = df[df['high_traffic'] == 'High']
    return df

def validate_numerical_values(df):
    # Ensure no negative values in nutritional columns
    nutritional_columns = ['calories', 'carbohydrate', 'sugar', 'protein']
    for col in nutritional_columns:
        df = df[df[col] >= 0]

    # Serving size validation (typically 1-6 servings)
    df = df[df['servings'].between(1, 6)]

    return df


def preprocess_data(df):
    # First correct data types
    df = correct_data_types(df)
    
    # Then handle missing values and duplicates
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    
    # Then validate data
    df = validate_categories(df)
    df = validate_high_traffic(df)
    df = validate_numerical_values(df)
    
    # Final type enforcement
    final_dtypes = {
        'recipe': 'int64',
        'calories': 'float64',
        'carbohydrate': 'float64',
        'sugar': 'float64',
        'protein': 'float64',
        'category': 'object',
        'servings': 'float64',
        'high_traffic': 'object'
    }
    
    df = df.astype(final_dtypes)
    
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
