import streamlit as st
import pandas as pd
import joblib
import pickle

# # Load the trained model
# model = joblib.load('heart_disease_model.joblib')

# Load the trained model
with open('chd_model_knn_classifier.pkl', 'rb') as file:
    model = pickle.load(file)



# Define the Streamlit app
def main():
    st.title("Heart Disease Prediction App")
    st.markdown("""
    This application uses a machine learning model to predict the likelihood of heart disease.
    Please enter the patient's details below:
    """)

    # Create input fields
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    st.text("Gender: ")
    sex_0 = st.checkbox("Female")
    sex_1 = st.checkbox("Male")
    st.write("chest pain type :")
    cp_0 = st.checkbox("Typical Angina")
    cp_1 = st.checkbox("Atypical Angina")
    cp_2 = st.checkbox("Non-anginal")
    cp_3 = st.checkbox("Asymptomatic")
    st.write("Resting Blood Pressure (mm Hg):")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
    st.write("Serum Cholesterol (mg/dl):")
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=1000, value=200)
    st.write("Fasting Blood Sugar > 120 mg/dl:")
    fbs_0 = st.checkbox("No Fasting Blood Sugar > 120 mg/dl")
    fbs_1 = st.checkbox("Yes Fasting Blood Sugar > 120 mg/dl")
    st.write("Resting Electrocardiographic Results:")
    restecg_0 = st.checkbox("Normal Resting Electrocardiographic Results")
    restecg_1 = st.checkbox("ST-T Abnormality Resting Electrocardiographic Results")
    restecg_2 = st.checkbox("LV Hypertrophy Resting Electrocardiographic Results")
    st.write("Maximum Heart Rate Achieved:")
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=150)
    st.write("Exercise-Induced Angina:")
    exang_0 = st.checkbox("No Exercise-Induced Angina")
    exang_1 = st.checkbox("Yes Exercise-Induced Angina")
    st.write("ST Depression Induced by Exercise Relative to Rest:")
    oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=1.0)
    st.write("Slope of the Peak Exercise ST Segment:")
    slope_0 = st.checkbox("Upsloping Slope of the Peak Exercise ST Segment")
    slope_1 = st.checkbox("Flat Slope of the Peak Exercise ST Segment")
    slope_2 = st.checkbox("Downsloping Slope of the Peak Exercise ST Segment")
    st.write("Number of Major Vessels Colored by Fluoroscopy:")
    ca_0 = st.checkbox("0 Major Vessels Colored by Fluoroscopy")
    ca_1 = st.checkbox("1 Major Vessel Colored by Fluoroscopy")
    ca_2 = st.checkbox("2 Major Vessels Colored by Fluoroscopy")
    ca_3 = st.checkbox("3 Major Vessels Colored by Fluoroscopy")
    ca_4 = st.checkbox("4 Major Vessels Colored by Fluoroscopy")
    st.write("Thalassemia:")
    thal_0 = st.checkbox("Normal Thalassemia")
    thal_1 = st.checkbox("Fixed Defect Thalassemia")
    thal_2 = st.checkbox("Reversible Defect Thalassemia")
    thal_3 = st.checkbox("Unknown Thalassemia")

    
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'age': [age],
        'trestbps': [trestbps],
        'chol': [chol],
        'thalach': [thalach],
        'oldpeak': [oldpeak],
        'sex_0': [sex_0],
        'sex_1': [sex_1],
        'cp_0': [cp_0],
        'cp_1': [cp_1],
        'cp_2': [cp_2],
        'cp_3': [cp_3],
        'fbs_0': [fbs_0],
        'fbs_1': [fbs_1],
        'restecg_0': [restecg_0],
        'restecg_1': [restecg_1],
        'restecg_2': [restecg_2],
        'exang_0': [exang_0],
        'exang_1': [exang_1],
        'slope_0': [slope_0],
        'slope_1': [slope_1],
        'slope_2': [slope_2],
        'ca_0': [ca_0],
        'ca_1': [ca_1],
        'ca_2': [ca_2],
        'ca_3': [ca_3],
        'ca_4': [ca_4],
        'thal_0': [thal_0],
        'thal_1': [thal_1],
        'thal_2': [thal_2],
        'thal_3': [thal_3]
    })
    
    # Display the input data
    st.write("### Input Data")
    st.write(input_data)

    # Make the prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        st.write(f"### Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
        st.write(f"### Probability of Heart Disease: {prediction_proba * 100:.2f}%")

if __name__ == "__main__":
    main()
