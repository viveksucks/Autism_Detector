from scipy.io import arff
import pandas as pd


data, meta = arff.loadarff("data/Autism-Adult-Data.arff")

df = pd.DataFrame(data)

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

questions = [
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

print("Question score distributions:\n")

for question in questions:
    print(question)
    print(df[question].value_counts())
    print()