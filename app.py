import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Prediction System")

st.markdown("### 🧠 AI-based Health Risk Prediction System")
st.markdown("Enter patient health data to predict diabetes risk using Machine Learning.")

st.subheader("Enter Patient Details")

st.divider()

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 0, 120)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Diabetes ({probability[0][1]*100:.2f}% confidence)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({probability[0][0]*100:.2f}% confidence)")