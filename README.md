# ASD Screening Tool

A machine learning project that predicts ASD screening outcomes using
behavioral questionnaire responses and basic demographic information.

The project uses XGBoost for prediction, SHAP for explainability, and
Streamlit for the web application.

> Note: This project is for educational purposes and is not a medical
> diagnostic tool.


## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Streamlit
- Joblib
- Git & GitHub

## Machine Learning

The model uses:

- Data preprocessing
- Feature analysis
- Cross-validation
- Hyperparameter tuning
- XGBoost classification
- Probability calibration
- SHAP explainability

Final XGBoost parameters:

```text
learning_rate = 0.2
max_depth = 3
n_estimators = 150
