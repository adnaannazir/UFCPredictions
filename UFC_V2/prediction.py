import joblib
import pandas as pd

# Load the model and scaler from the provided files
model = joblib.load('fighter_prediction_model.pkl')
scaler = joblib.load('fighter_prediction_scaler.pkl')


def predict_fight_outcome(fighter1_name, fighter2_name):
    # Load the dataset
    data = pd.read_csv('FinalCleanedData.csv')
    
    # Function to parse fighter records into wins, losses, and ties
    def parse_record(record):
        parts = record.split('-')
        wins = int(parts[0])
        losses = int(parts[1].split(' ')[0])  # Modify to split and take the first part only
        ties = int(parts[2].split(' ')[0]) if len(parts) > 2 else 0
        return wins, losses, ties
    
    # Calculate derived features for the dataset
    data[['Wins', 'Losses', 'Ties']] = data['Overall Record'].apply(
        lambda x: pd.Series(parse_record(x))
    )
    data['Win-Loss Ratio'] = data['Wins'] / (data['Losses'] + 1)
    data['Strike Differential'] = data['Sig. Strikes Landed/min'] - data['Sig. Strikes Absorbed/min']
    data['Experience'] = data['Wins'] + data['Losses'] + data['Ties']
    
    # Find the data for both fighters
    fighter1_data = data[data['Fighter Name'] == fighter1_name]
    fighter2_data = data[data['Fighter Name'] == fighter2_name]
    
    # If either fighter is not found, return an error message
    if fighter1_data.empty or fighter2_data.empty:
        missing_fighters = []
        if fighter1_data.empty:
            missing_fighters.append(fighter1_name)
        if fighter2_data.empty:
            missing_fighters.append(fighter2_name)
        return f"Fighter(s) {', '.join(missing_fighters)} not found in the dataset."
    
    # Features as defined in training with suffixes _1 and _2 for each fighter
    feature_columns = [
        'Age', 'Height', 'Reach', 'Wins', 'Losses', 'Ties', 'Win-Loss Ratio',
        'Sig. Strikes Landed/min', 'Sig. Strikes Absorbed/min', 'Strike Differential',
        'Striking Accuracy (%)', 'Striking Defense (%)', 'Experience', 'Odds'
    ]
    features_1 = {f"{feat}_1": fighter1_data[feat].values[0] for feat in feature_columns}
    features_2 = {f"{feat}_2": fighter2_data[feat].values[0] for feat in feature_columns}
    
    # Create combined feature array
    combined_features = list(features_1.values()) + list(features_2.values())
    
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform([combined_features])
    
    # Predict using the model
    prediction = model.predict(scaled_features)
    
    # Return the winner based on the prediction result
    if prediction[0] == 0:
        return fighter2_name  # Fighter 2 is predicted to win
    else:
        return fighter1_name  # Fighter 1 is predicted to win


