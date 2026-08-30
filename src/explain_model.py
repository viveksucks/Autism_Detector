from scipy.io import arff
import pandas as pd
import shap
import matplotlib.pyplot as plt

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

df.loc[df["age"] > 100, "age"] = pd.NA

df["ethnicity"] = df["ethnicity"].replace("others", "Others")

X = df.drop(columns=["Class/ASD", "result", "age_desc"])
y = df["Class/ASD"]

y = y.map({"NO": 0, "YES": 1})

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

pipeline.fit(X, y)

X_transformed = pipeline.named_steps["preprocessor"].transform(X)

model = pipeline.named_steps["model"]

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_transformed)

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.savefig("results/shap_summary.png", bbox_inches="tight")

plt.close()

print("SHAP plot saved successfully.")