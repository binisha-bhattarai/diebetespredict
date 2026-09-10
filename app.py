import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load(
    "models/diabetes_final_model.pkl"
)

scaler = joblib.load(
    "models/diabetes_final_scaler.pkl"
)
st.title(" Health")
st.subheader("Diabetes Risk Prediction")

st.write(
    "Enter patient information to generate a machine-learning prediction."
)

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

if st.button("Predict Diabetes Risk"):

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.write("Prediction probability:", round(probability * 100, 2), "%")

    if prediction == 1:
        st.warning("The model predicts a higher diabetes-risk class.")
    else:
        st.success("The model predicts a lower diabetes-risk class.")

st.divider()

st.header("📊 Diabetes Dataset Analytics")

data = pd.read_csv("data/diabetes.csv")

st.write(
    "Dataset overview"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Records",
        len(data)
    )

with col2:
    st.metric(
        "Features",
        data.shape[1] - 1
    )

with col3:
    st.metric(
        "Positive Outcomes",
        int(data["Outcome"].sum())
    )
    st.subheader(
    "Diabetes Outcome Distribution"
)

outcome_counts = data["Outcome"].value_counts()

st.bar_chart(
    outcome_counts
)
st.subheader(
    "Glucose Distribution"
)

st.bar_chart(
    data["Glucose"].value_counts().sort_index()
)
with st.expander(
    "View Dataset"
):

    st.dataframe(
        data,
        use_container_width=True
    )
st.divider()

st.header(
    "🤖 Model Information"
)



st.write(
    "The system uses supervised machine learning "
    "for binary classification."
)

st.write(
    "The model receives patient measurements and "
    "predicts the class represented by the dataset's "
    "Outcome variable."
)
st.write(
    "Final model: Tuned Random Forest"
)