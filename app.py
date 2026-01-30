import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
train_df = pd.read_csv("Training.csv")

# Remove unwanted columns
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]

# Split features and target
X = train_df.drop("prognosis", axis=1)
y = train_df["prognosis"]

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Prediction function
def predict_disease(symptom_text, min_symptoms=3):
    user_symptoms = [
        s.strip().lower().replace(" ", "_").replace("'", "").replace('"', "")
        for s in symptom_text.split(",")
    ]

    if len(user_symptoms) < min_symptoms:
        return "Please enter at least 3 symptoms."

    input_dict = {symptom: 0 for symptom in X.columns}

    matched = 0
    for symptom in user_symptoms:
        if symptom in input_dict:
            input_dict[symptom] = 1
            matched += 1

    if matched < min_symptoms:
        return "Symptoms not recognized properly."

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)
    return prediction[0]

# ---------- STREAMLIT UI ----------
st.title("🩺 Smart Disease Prediction System")

st.write("Enter symptoms separated by commas:")

user_input = st.text_input(
    "Example: itching, skin rash, nodal skin eruptions"
)

if st.button("Predict Disease"):
    if user_input.strip() == "":
        st.warning("Please enter symptoms.")
    else:
        result = predict_disease(user_input)
        st.success(f"Predicted Disease: {result}")

