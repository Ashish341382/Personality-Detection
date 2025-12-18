import pickle
import streamlit as st
import pandas as pd

# -------------------------------
# Load model and scaler
# -------------------------------
with open('personality_log.pkl', 'rb') as f:
    log_model = pickle.load(f)

with open('personality_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# -------------------------------
# App config
# -------------------------------
st.set_page_config(page_title="Personality Predictor", layout="centered")
st.title("Personality Type Prediction")
st.write("Provide your personality traits to predict your personality type.")

# -------------------------------
# Feature list (MUST match training)
# -------------------------------
features = [
    'social_energy', 'alone_time_preference', 'talkativeness',
    'deep_reflection', 'group_comfort', 'party_liking',
    'listening_skill', 'empathy', 'organization',
    'leadership', 'risk_taking', 'public_speaking_comfort', 'curiosity',
    'routine_preference', 'excitement_seeking', 'friendliness',
    'planning', 'spontaneity', 'adventurousness',
    'reading_habit', 'sports_interest', 'online_social_usage',
    'travel_desire', 'gadget_usage', 'work_style_collaborative',
    'decision_speed'
]

# -------------------------------
# User input section (2 sliders per row)
# -------------------------------
st.subheader("Enter Feature Values")

input_data = {}

for i in range(0, len(features), 2):
    col1, col2 = st.columns(2)

    with col1:
        input_data[features[i]] = st.slider(
            features[i].replace("_", " ").title(),
            min_value=0.0,
            max_value=10.0,
            value=5.0
        )

    if i + 1 < len(features):
        with col2:
            input_data[features[i + 1]] = st.slider(
                features[i + 1].replace("_", " ").title(),
                min_value=0.0,
                max_value=10.0,
                value=5.0
            )

# Convert to DataFrame
input_df = pd.DataFrame([input_data])



label_map = {
    0: "Introvert",
    1: "Ambivert",
    2: "Extrovert"
}


# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Personality"):
    input_scaled = scaler.transform(input_df)
    
    label_map = {
    0: "Introvert",
    1: "Ambivert",
    2: "Extrovert"
    
    }
    
    pred = log_model.predict(input_scaled)[0]
    personality = label_map.get(pred, "Unknown")

    st.success(f"Predicted Personality Type: **{personality}**")


    
    
    