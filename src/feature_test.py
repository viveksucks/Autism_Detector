from scipy.io import arff
import pandas as pd

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

y = df["Class/ASD"].map({
    "NO": 0,
    "YES": 1
})


def test_model(features, name):

    X = df[features].copy()

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

    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.2,
        random_state=42,
        eval_metric="logloss"
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    print("\n" + name)
    print("-" * len(name))

    print(
        "Accuracy:",
        scores["test_accuracy"].mean()
    )

    print(
        "Precision:",
        scores["test_precision"].mean()
    )

    print(
        "Recall:",
        scores["test_recall"].mean()
    )

    print(
        "F1 Score:",
        scores["test_f1"].mean()
    )


all_features = [
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
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation"
]

without_austim = [
    feature for feature in all_features
    if feature != "austim"
]

test_model(
    all_features,
    "Model with austim"
)

test_model(
    without_austim,
    "Model without austim"
)