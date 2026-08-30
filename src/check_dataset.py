from scipy.io import arff
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
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

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Final Model Results")
print("-------------------")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))