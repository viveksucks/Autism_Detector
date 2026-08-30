# ASD Screening Tool

A machine learning project that uses behavioral screening responses and
basic demographic information to classify ASD screening outcomes.

The project includes the complete ML workflow — from dataset analysis and
preprocessing to model training, hyperparameter tuning, probability
calibration, explainability, and an interactive Streamlit application.

> **Note:** This is an academic/portfolio project and is not intended for
> medical diagnosis or clinical decision-making.

---

## Overview

The goal of this project was to build a complete end-to-end machine learning
system rather than just train a classifier and report its accuracy.

The workflow covers:

- Dataset exploration and cleaning
- Feature analysis
- Numerical and categorical preprocessing
- Train/test splitting
- Cross-validation
- XGBoost model development
- Hyperparameter tuning
- Feature testing
- Probability calibration
- Model evaluation
- SHAP-based explainability
- Interactive Streamlit application

The final model achieved **99.29% accuracy** and an **F1 score of 98.67%**
on the held-out test set.

---

## Demo

The project includes an interactive Streamlit application where a user can:

1. Answer the 10 behavioral screening questions
2. Enter the required personal information
3. View the screening score
4. Get the model prediction
5. See the predicted probabilities
6. View the features that influenced the prediction through SHAP

### Application

```text
Questionnaire
      ↓
Input preprocessing
      ↓
Calibrated XGBoost model
      ↓
Prediction + probability
      ↓
SHAP explanation