from src.data_preprocessing import load_and_clean_data
from src.model import train_model, save_model
from src.evaluation import evaluate_model

def main():
    # Load and clean data
    df = load_and_clean_data()
    
    # Train model
    model = train_model(df)
    
    # Evaluate model
    metrics = evaluate_model(model, df)
    
    # Print or save metrics
    print(metrics)

if __name__ == "__main__":
    main()
