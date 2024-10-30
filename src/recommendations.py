import pandas as pd

def predict_traffic_potential(recipe_data):
    """
    Predict traffic potential for a recipe based on key characteristics
    """
    # Score components
    score = 0
    max_score = 100
    feedback = []
    
    # 1. Calorie Content (45% importance)
    calorie_score = 0
    if 300 <= recipe_data['calories'] <= 600:
        calorie_score = 45  # Optimal range
    elif 200 <= recipe_data['calories'] <= 800:
        calorie_score = 30  # Good range
    else:
        calorie_score = 15  # Outside optimal range
    
    score += calorie_score
    feedback.append(f"Calorie Score: {calorie_score}/45")
    
    # 2. Protein Content (42% importance)
    protein_score = 0
    if recipe_data['protein'] >= 20:
        protein_score = 42  # High protein
    elif recipe_data['protein'] >= 10:
        protein_score = 28  # Moderate protein
    else:
        protein_score = 14  # Low protein
    
    score += protein_score
    feedback.append(f"Protein Score: {protein_score}/42")
    
    # 3. Serving Size (12% importance)
    serving_score = 0
    if recipe_data['servings'] == 4:
        serving_score = 12  # Optimal servings
    elif 2 <= recipe_data['servings'] <= 6:
        serving_score = 8   # Good range
    else:
        serving_score = 4   # Outside optimal range
    
    score += serving_score
    feedback.append(f"Serving Score: {serving_score}/12")
    
    # Traffic potential classification
    if score >= 85:
        traffic_potential = "Very High"
    elif score >= 70:
        traffic_potential = "High"
    elif score >= 50:
        traffic_potential = "Moderate"
    else:
        traffic_potential = "Low"
    
    return {
        'traffic_potential': traffic_potential,
        'score': score,
        'feedback': feedback,
        'recommendations': generate_recommendations(recipe_data, score)
    }

def generate_recommendations(recipe_data, score):
    """
    Generate specific recommendations to improve traffic potential
    """
    recommendations = []
    
    # Calorie recommendations
    if recipe_data['calories'] < 300:
        recommendations.append("Consider increasing calories to 300-600 range")
    elif recipe_data['calories'] > 600:
        recommendations.append("Consider reducing calories to 300-600 range")
    
    # Protein recommendations
    if recipe_data['protein'] < 20:
        recommendations.append("Increase protein content to at least 20g")
    
    # Serving recommendations
    if recipe_data['servings'] != 4:
        recommendations.append("Adjust serving size to 4 portions")
    
    return recommendations

def analyze_recipe_portfolio(df):
    """
    Analyze entire recipe portfolio for traffic potential
    """
    results = []
    for _, recipe in df.iterrows():
        prediction = predict_traffic_potential(recipe)
        results.append({
            'recipe_id': recipe['recipe'],
            'category': recipe['category'],
            'traffic_potential': prediction['traffic_potential'],
            'score': prediction['score']
        })
    
    return pd.DataFrame(results)


def main():
    # Load data
    df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # Analyze entire portfolio
    portfolio_analysis = analyze_recipe_portfolio(df)
    
    # Print high-traffic recipes
    high_traffic = portfolio_analysis[portfolio_analysis['traffic_potential'].isin(['High', 'Very High'])]
    
    print("\nHigh Traffic Recipe Characteristics:")
    print("====================================")
    
    # Group by category
    category_stats = high_traffic.groupby('category').size().sort_values(ascending=False)
    print("\nHigh Traffic Recipes by Category:")
    print(category_stats)
    
    # Print example recipes from original dataset
    print("\nExample High-Traffic Recipes:")
    high_traffic_recipes = df[df['recipe'].isin(high_traffic['recipe_id'])]
    sample_recipes = high_traffic_recipes.sample(5)
    
    for _, recipe in sample_recipes.iterrows():
        prediction = predict_traffic_potential(recipe)
        print(f"\nRecipe {recipe['recipe']}")
        print(f"Category: {recipe['category']}")
        print(f"Calories: {recipe['calories']:.0f}")
        print(f"Protein: {recipe['protein']:.1f}g")
        print(f"Servings: {recipe['servings']}")
        print(f"Traffic Potential: {prediction['traffic_potential']}")
        print(f"Score: {prediction['score']}")
        print("Recommendations:")
        for rec in prediction['recommendations']:
            print(f"- {rec}")

if __name__ == "__main__":
    main()