import pandas as pd
import joblib


model_data = joblib.load("models/autism_model.pkl")

model = model_data["model"]


base_data = {
    "age": 25,
    "gender": "m",
    "ethnicity": "White-European",
    "jundice": "no",
    "contry_of_res": "India",
    "used_app_before": "no",
    "relation": "Self"
}


for score in range(0, 11):

    case = {}

    for i in range(1, 11):
        case[f"A{i}_Score"] = 1 if i <= score else 0

    case.update(base_data)

    input_data = pd.DataFrame([case])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    print(
        f"{score}/10 -> "
        f"{'YES' if prediction == 1 else 'NO'} "
        f"({probability:.2%} YES)"
    )