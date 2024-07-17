import streamlit as st
import numpy as np
import pickle

with open('chd_model_knn_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

def main():
    st.title("Heart Disease Prediction")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)

    sex = st.radio("Sex", options=["Male", "Female"])
    cp = st.radio("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
    restecg = st.radio("Resting Electrocardiographic Results", options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    exang = st.radio("Exercise Induced Angina", options=["Yes", "No"])
    slope = st.radio("Slope of the Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
    ca = st.radio("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
    thal = st.radio("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])

    # Convert categorical variables to boolean values
    sex_0 = 1 if sex == "Male" else 0
    sex_1 = 0 if sex == "Male" else 1
    cp_0 = 1 if cp == "Typical Angina" else 0
    cp_1 = 1 if cp == "Atypical Angina" else 0
    cp_2 = 1 if cp == "Non-anginal" else 0
    cp_3 = 1 if cp == "Asymptomatic" else 0
    fbs_0 = 1 if fbs == "No" else 0
    fbs_1 = 1 if fbs == "Yes" else 0
    restecg_0 = 1 if restecg == "Normal" else 0
    restecg_1 = 1 if restecg == "ST-T wave abnormality" else 0
    restecg_2 = 1 if restecg == "Left ventricular hypertrophy" else 0
    exang_0 = 1 if exang == "No" else 0
    exang_1 = 1 if exang == "Yes" else 0
    slope_0 = 1 if slope == "Upsloping" else 0
    slope_1 = 1 if slope == "Flat" else 0
    slope_2 = 1 if slope == "Downsloping" else 0
    ca_0 = 1 if ca == 0 else 0
    ca_1 = 1 if ca == 1 else 0
    ca_2 = 1 if ca == 2 else 0
    ca_3 = 1 if ca == 3 else 0
    ca_4 = 0  # Assuming no other values for 'ca'
    thal_0 = 1 if thal == "Normal" else 0
    thal_1 = 1 if thal == "Fixed Defect" else 0
    thal_2 = 1 if thal == "Reversible Defect" else 0
    thal_3 = 0  # Assuming no other values for 'thal'

    # Create input data array
    input_data = np.array([[age, trestbps, chol, thalach, oldpeak, sex_0, sex_1, cp_0, cp_1, cp_2, cp_3,
                            fbs_0, fbs_1, restecg_0, restecg_1, restecg_2, exang_0, exang_1, slope_0,
                            slope_1, slope_2, ca_0, ca_1, ca_2, ca_3, ca_4, thal_0, thal_1, thal_2, thal_3]])

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        st.write(f"### Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
        st.write(f"### Probability of Heart Disease: {prediction_proba * 100:.2f}%")

if __name__ == "__main__":
    main()
