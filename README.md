🏥 AI Healthcare Analytics & Prediction Dashboard

Machine learning–powered diabetes risk prediction, wrapped in an interactive Streamlit dashboard.

Enter a patient's health measurements and get an instant, ML-backed diabetes risk prediction and live analytics on the underlying dataset.

bash
streamlit run app.py
📊 Dataset

Pima Indians Diabetes Dataset

	
Records	768
Features	8
Target	Outcome (0 = non-diabetic, 1 = diabetic)
Class balance	500 negative / 268 positive

⚠️ Glucose, BloodPressure, SkinThickness, Insulin, and BMI use 0 as a missing-value placeholder a known quirk of this dataset, handled during preprocessing.

🔬 Approach

1. EDA & Preprocessing — distribution analysis, correlation heatmap, stratified 80/20 train/test split, StandardScaler feature scaling.

2. Model Comparison — trained and benchmarked 6 classifiers on identical data splits:

Model	Accuracy	Precision	Recall	F1	ROC-AUC
🥇 Gradient Boosting	0.7532	0.6667	0.5926	0.6275	0.8407
Logistic Regression	0.7143	0.6087	0.5185	0.5600	0.8230
Random Forest	0.7597	0.6809	0.5926	0.6337	0.8147
SVM	0.7532	0.6600	0.6111	0.6346	0.7924
KNN	0.7013	0.5833	0.5185	0.5490	0.7405
Decision Tree	0.7208	0.6341	0.4815	0.5474	0.6657

3. Hyperparameter Tuning — tuned a RandomForestClassifier (max_depth=5, min_samples_split=5, n_estimators=100), which became the production model.

4. Deployment — served via a Streamlit app with real-time prediction and dataset analytics.

🏆 Results

Final production model: Tuned Random Forest

Metric	Score
Accuracy	TODO
Precision	TODO
Recall	TODO
F1 Score	TODO
ROC-AUC	TODO
🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib · Seaborn · Joblib

📁 Project Structure
├── app.py                            # Streamlit app
├── data/
│   └── diabetes.csv
├── models/
│   ├── diabetes_best_model.pkl       # Gradient Boosting (initial comparison winner)
│   ├── diabetes_scaler.pkl
│   ├── diabetes_logistic_model.pkl   # Baseline logistic regression
│   ├── diabetes_final_model.pkl      # Tuned Random Forest (production)
│   └── diabetes_final_scaler.pkl
├── notebooks/
│   ├── 01_diabetes_eda.ipynb         # EDA, preprocessing, model comparison
│   └── 02_diabetes_model_training.ipynb  # Hyperparameter tuning
├── test_diabetes_model.py
├── requirements.txt
└── README.md
