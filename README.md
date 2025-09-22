Heart Disease Prediction using Machine Learning
1. Introduction

Heart disease is a leading cause of mortality worldwide.

Early detection can help reduce complications and healthcare costs.

This project applies machine learning to predict the likelihood of heart disease based on clinical data.

2. Problem Statement

Predict whether a patient is at risk of heart disease (binary classification).

Dataset includes parameters such as age, sex, blood pressure, cholesterol, fasting blood sugar, ECG results, chest pain type, and exercise angina.

Goal: Build accurate, interpretable, and scalable models to assist in preventive healthcare.

3. Dataset

Source: UCI Heart Disease Dataset (via Kaggle)

Total records: 918 patients

Features: 11+ clinical indicators (numerical + categorical)

Balanced dataset: ~55% with disease, ~45% without disease

4. Exploratory Data Analysis (EDA)

4.1 Checked for missing values (none found).
4.2 Analyzed distributions of age, cholesterol, blood pressure, etc.
4.3 Key insights:

Age group 56–70 had the highest risk (71%).

Males showed higher incidence (63%) compared to females (25%).

Chest pain type “ASY” had ~79% disease presence.

ECG and ST slope were strong predictors.

5. Data Preprocessing

5.1 Standardized numerical features using StandardScaler.
5.2 Encoded categorical features with OneHotEncoder.
5.3 Created derived features (e.g., mapping ExerciseAngina, ST_Slope, Oldpeak into binary).
5.4 Split dataset into 80% training, 20% testing.

6. Models Used

Logistic Regression

Random Forest

K-Nearest Neighbors (KNN)

Gradient Boosting

Hyperparameter tuning performed with GridSearchCV.

7. Results
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	83%	0.90	0.80	0.85
Random Forest	88%	0.90	0.89	0.90
KNN	86%	0.88	0.88	0.88
Gradient Boosting	88%	0.92	0.87	0.89

Best model: Random Forest with 88% accuracy and F1-score of 0.90.

8. Visualizations

Distribution plots for numerical features

Bar charts showing categorical feature impact on disease risk

Correlation heatmap

Confusion matrices for each model

9. Future Work

Expand dataset with more diverse demographics.

Use advanced models (XGBoost, Neural Networks).

Deploy as a web or mobile app for real-time predictions.

Focus on interpretable ML (e.g., SHAP, LIME) for medical use cases.

10. Tech Stack

Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Tools: Jupyter Notebook
