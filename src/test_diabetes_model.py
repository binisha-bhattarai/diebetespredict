import pandas as pd
import joblib


# Load model and scaler

model = joblib.load(
    "models/diabetes_logistic_model.pkl"
)

scaler = joblib.load(
    "models/diabetes_scaler.pkl"
)


# Example patient

sample = pd.DataFrame({
    "Pregnancies": [1],
    "Glucose": [120],
    "BloodPressure": [70],
    "SkinThickness": [20],
    "Insulin": [80],
    "BMI": [25.0],
    "DiabetesPedigreeFunction": [0.5],
    "Age": [30]
})


# Scale input

sample_scaled = scaler.transform(sample)


# Prediction

prediction = model.predict(sample_scaled)[0]

probability = model.predict_proba(
    sample_scaled
)[0][1]


print("Prediction:", prediction)

print(
    "Probability:",
    round(probability * 100, 2),
    "%"
)