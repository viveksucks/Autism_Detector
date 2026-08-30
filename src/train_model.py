from scipy.io import arff
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


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
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["age"] = pd.to_numeric(df["age"], errors="coerce")

df.loc[df["age"] > 100, "age"] = pd.NA

df["ethnicity"] = df["ethnicity"].replace("others", "Others")


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

y = df["Class/ASD"].map({
    "NO": 0,
    "YES": 1
})


categorical_columns = X.select_dtypes(include="str").columns
numeric_columns = X.select_dtypes(exclude="str").columns


numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.2,
    random_state=42,
    eval_metric="logloss"
)


pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)


pipeline.fit(X, y)


calibrated_model = CalibratedClassifierCV(
    pipeline,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X, y)


model_data = {
    "model": calibrated_model,
    "base_model": pipeline,
    "features": features
}


joblib.dump(
    model_data,
    "models/autism_model.pkl"
)


print("Calibrated model saved successfully.")

print("\nFeatures used:")
print(features)