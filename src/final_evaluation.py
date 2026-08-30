from scipy.io import arff
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


data, meta = arff.loadarff("data/Autism-Adult-Data.arff")

df = pd.DataFrame(data)


for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )


df = df.replace("?", pd.NA)


score_columns = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score"
]


for column in score_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)


df.loc[df["age"] > 100, "age"] = pd.NA


df["ethnicity"] = df["ethnicity"].replace(
    "others",
    "Others"
)


features = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
    "age",
    "gender",
    "ethnicity",
    "jundice",
    "contry_of_res",
    "used_app_before",
    "relation"
]


X = df[features]

y = df["Class/ASD"].map(
    {
        "NO": 0,
        "YES": 1
    }
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model_data = joblib.load(
    "models/autism_model.pkl"
)

model = model_data["model"]


model.fit(
    X_train,
    y_train
)


predictions = model.predict(X_test)


print("\nFinal Model Evaluation")
print("----------------------")


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)


print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1 Score: {f1:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)