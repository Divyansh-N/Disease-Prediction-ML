import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train_df = pd.read_csv("Training.csv")


train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]


X = train_df.drop("prognosis", axis=1)
y = train_df["prognosis"]


model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)


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

st.write("Select symptoms from the list below:")


symptoms_list = list(X.columns)


selected_symptoms = st.multiselect(
    "Choose symptoms:",
    symptoms_list
)

if st.button("Predict Disease"):

    if len(selected_symptoms) < 3:
        st.warning("Please select at least 3 symptoms")

    else:
        
        input_dict = {symptom: 0 for symptom in X.columns}

        for symptom in selected_symptoms:
            input_dict[symptom] = 1

        
        input_df = pd.DataFrame([input_dict])

        
        probs = model.predict_proba(input_df)[0]

    
        top_indices = probs.argsort()[-3:][::-1]

        st.subheader("Possible Diseases:")

        for i in top_indices:
            disease = model.classes_[i]
            probability = probs[i] * 100
            st.write(f"{disease} — {probability:.2f}%")


