import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('xgb_fight_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load the data
clean_data = pd.read_csv('CleanData2.csv')

# Streamlit app
st.title('Fight Outcome Prediction')

# Dropdown to select fighters
fighter1 = st.selectbox('Select Fighter 1', clean_data['Fighter Name'].unique())
fighter2 = st.selectbox('Select Fighter 2', clean_data['Fighter Name'].unique())

if st.button('Predict Winner'):
    if fighter1 == fighter2:
        st.write("Please select two different fighters.")
    else:
        # Get the fighter data
        fighter1_data = clean_data[clean_data['Fighter Name'] == fighter1]
        fighter2_data = clean_data[clean_data['Fighter Name'] == fighter2]

        # Prepare the feature vector
        try:
            # Extract the features and compute the differences
            age_difference = fighter1_data['Age'].values[0] - fighter2_data['Age'].values[0]
            height_f1 = fighter1_data['Height'].values[0]
            height_f2 = fighter2_data['Height'].values[0]
            reach_f1 = fighter1_data['Reach'].values[0]
            reach_f2 = fighter2_data['Reach'].values[0]
            strikes_f1 = fighter1_data['Sig. Strikes Landed/min'].values[0]
            strikes_f2 = fighter2_data['Sig. Strikes Landed/min'].values[0]

            # Combine all features into a single feature vector
            features_vector = np.array([age_difference, height_f1, height_f2, reach_f1, reach_f2, strikes_f1, strikes_f2]).reshape(1, -1)

            # Scale the features
            X_new_scaled = scaler.transform(features_vector)

            # Make prediction
            prediction = model.predict(X_new_scaled)
            winner = fighter1 if prediction[0] == 1 else fighter2
            st.write(f"The predicted winner is: {winner}")
        except Exception as e:
            st.write("Error in prediction:", e)
