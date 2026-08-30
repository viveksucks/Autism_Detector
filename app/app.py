import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# LOAD MODEL
# ============================================================

model_data = joblib.load("models/autism_model.pkl")

model = model_data["model"]
base_model = model_data["base_model"]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ASD Screening Tool",
    page_icon="🧩",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] *,
    [data-testid="stHeader"],
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        font-family: "Manrope", "Segoe UI", sans-serif !important;
    }

    .block-container {
        max-width: 1200px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 5rem !important;
    }

    h1 {
        font-family: "Manrope", sans-serif !important;
        font-size: 50px !important;
        font-weight: 800 !important;
        letter-spacing: -1.8px !important;
        line-height: 1.15 !important;
    }

    h2 {
        font-family: "Manrope", sans-serif !important;
        font-size: 32px !important;
        font-weight: 750 !important;
        letter-spacing: -0.8px !important;
        line-height: 1.25 !important;
        margin-top: 35px !important;
    }

    h3 {
        font-family: "Manrope", sans-serif !important;
        font-size: 25px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMarkdownContainer"] p {
        font-family: "Manrope", sans-serif !important;
        font-size: 17px !important;
        line-height: 1.65 !important;
    }

    div[data-testid="stRadio"] > label {
        font-family: "Manrope", sans-serif !important;
        font-size: 22px !important;
        font-weight: 650 !important;
        line-height: 1.55 !important;
        margin-bottom: 8px !important;
    }

    div[data-testid="stRadio"] > label p {
        font-family: "Manrope", sans-serif !important;
        font-size: 22px !important;
        font-weight: 650 !important;
        line-height: 1.55 !important;
    }

    div[data-testid="stRadio"]
    [role="radiogroup"]
    label {
        font-family: "Manrope", sans-serif !important;
        font-size: 18px !important;
        font-weight: 550 !important;
    }

    div[data-testid="stRadio"]
    [role="radiogroup"]
    label p {
        font-family: "Manrope", sans-serif !important;
        font-size: 18px !important;
        font-weight: 550 !important;
    }

    .question-number {
        font-family: "Manrope", sans-serif !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px;
        opacity: 0.65;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label {
        font-family: "Manrope", sans-serif !important;
        font-size: 17px !important;
        font-weight: 650 !important;
    }

    input {
        font-family: "Manrope", sans-serif !important;
        font-size: 17px !important;
    }

    div[data-baseweb="select"] {
        font-family: "Manrope", sans-serif !important;
        font-size: 17px !important;
    }

    div.stButton > button {
        font-family: "Manrope", sans-serif !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        min-height: 56px !important;
        border-radius: 11px !important;
    }

    [data-testid="stMetricValue"] {
        font-family: "Manrope", sans-serif !important;
        font-size: 38px !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: "Manrope", sans-serif !important;
        font-size: 15px !important;
        font-weight: 650 !important;
    }

    [data-testid="stAlert"] p {
        font-family: "Manrope", sans-serif !important;
        font-size: 16px !important;
        line-height: 1.55 !important;
    }

    [data-testid="stProgress"] {
        margin-top: 10px !important;
        margin-bottom: 25px !important;
    }

    hr {
        margin-top: 35px !important;
        margin-bottom: 35px !important;
        opacity: 0.25;
    }

    [data-testid="stExpander"] summary {
        font-family: "Manrope", sans-serif !important;
        font-size: 16px !important;
        font-weight: 650 !important;
    }

    .footer {
        text-align: center;
        opacity: 0.45;
        font-size: 14px;
        line-height: 1.7;
        padding-top: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🧩 ASD Screening Tool")

st.markdown(
    "### Machine learning based behavioral screening"
)

st.write(
    "This application analyzes responses from a 10-question "
    "behavioral screening questionnaire together with selected "
    "personal information. A calibrated XGBoost model generates "
    "the screening prediction, while SHAP provides an explanation "
    "of the factors that influenced the model's decision."
)

st.info(
    "This project is intended for educational and research purposes. "
    "It is not a medical diagnostic tool."
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.header("Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Screening Questions",
        "10"
    )
    st.caption(
        "Behavioral questionnaire features"
    )

with col2:
    st.metric(
        "ML Algorithm",
        "XGBoost"
    )
    st.caption(
        "Calibrated classification model"
    )

with col3:
    st.metric(
        "Explainability",
        "SHAP"
    )
    st.caption(
        "Feature-level model explanation"
    )


st.divider()


# ============================================================
# SCREENING QUESTIONNAIRE
# ============================================================

st.header("1. Screening Questionnaire")

st.write(
    "Answer each statement based on your usual behavior."
)


questions = [

    (
        "A1_Score",
        "I often notice small sounds when others do not.",
        True
    ),

    (
        "A2_Score",
        "I usually concentrate more on the whole picture rather than the small details.",
        False
    ),

    (
        "A3_Score",
        "I find it easy to do more than one thing at once.",
        False
    ),

    (
        "A4_Score",
        "If there is an interruption, I can switch back to what I was doing very quickly.",
        False
    ),

    (
        "A5_Score",
        "I find it easy to read between the lines when someone is talking to me.",
        False
    ),

    (
        "A6_Score",
        "I know how to tell if someone listening to me is getting bored.",
        False
    ),

    (
        "A7_Score",
        "When I am reading a story, I find it difficult to work out the characters' intentions.",
        True
    ),

    (
        "A8_Score",
        "I like to collect information about categories of things.",
        True
    ),

    (
        "A9_Score",
        "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
        False
    ),

    (
        "A10_Score",
        "I find it difficult to work out people's intentions.",
        True
    )

]


answers = {}


for number, (column, question, yes_is_one) in enumerate(
    questions,
    start=1
):

    st.markdown(
        f'<div class="question-number">QUESTION {number:02d}</div>',
        unsafe_allow_html=True
    )

    answer = st.radio(
        question,
        ["No", "Yes"],
        horizontal=True,
        key=column
    )

    if yes_is_one:

        answers[column] = (
            1 if answer == "Yes" else 0
        )

    else:

        answers[column] = (
            1 if answer == "No" else 0
        )

    if number != 10:
        st.write("")


# ============================================================
# SCREENING SCORE
# ============================================================

screening_score = sum(
    answers.values()
)


st.divider()


score_col1, score_col2 = st.columns(
    [1, 3]
)


with score_col1:

    st.metric(
        "Screening Score",
        f"{screening_score}/10"
    )


with score_col2:

    st.write(
        "**Screening Progress**"
    )

    st.progress(
        screening_score / 10
    )


st.divider()


# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.header("2. Personal Information")

st.write(
    "Additional information used by the machine learning model."
)


col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25,
        step=1
    )

    st.caption(
        "Model-supported age range: 17–64"
    )

    gender = st.selectbox(
        "Gender",
        [
            "m",
            "f"
        ]
    )


with col2:

    ethnicity = st.selectbox(
        "Ethnicity",
        [
            "White-European",
            "Asian",
            "Middle Eastern",
            "Black",
            "South Asian",
            "Others",
            "Latino",
            "Hispanic",
            "Pasifika",
            "Turkish",
            "Unknown"
        ]
    )

    jundice = st.selectbox(
        "History of jaundice",
        [
            "no",
            "yes"
        ]
    )


with col3:

    country = st.text_input(
        "Country of residence",
        "India"
    )

    used_app_before = st.selectbox(
        "Used a screening application before?",
        [
            "no",
            "yes"
        ]
    )


relation = st.selectbox(
    "Who completed the screening?",
    [
        "Self",
        "Parent",
        "Relative",
        "Others",
        "Health care professional",
        "Unknown"
    ]
)


st.write("")

st.divider()


# ============================================================
# RUN SCREENING
# ============================================================

st.header("3. Run Screening")

st.write(
    "Review your answers and run the trained machine learning model."
)


if st.button(
    "Run Screening",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # AGE VALIDATION
    # --------------------------------------------------------

    if age < 17 or age > 64:

        st.error(
            "This model was trained on screening records for "
            "ages 17–64. A prediction cannot be generated for "
            "this age."
        )

        st.stop()


    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                **answers,

                "age": age,

                "gender": gender,

                "ethnicity": ethnicity,

                "jundice": jundice,

                "contry_of_res": country,

                "used_app_before": used_app_before,

                "relation": relation
            }
        ]
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    probabilities = model.predict_proba(
        input_data
    )[0]


    probability_no = probabilities[0]

    probability_yes = probabilities[1]


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.divider()

    st.header("Screening Result")


    if prediction == 1:

        st.error(
            "Positive screening result"
        )

        result_text = (
            "The model classified this input as YES based on "
            "patterns learned from the training data."
        )

    else:

        st.success(
            "Negative screening result"
        )

        result_text = (
            "The model classified this input as NO based on "
            "patterns learned from the training data."
        )


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Model Confidence",
            f"{probability_yes:.1%}"
        )


    with result_col2:

        st.metric(
            "Screening Score",
            f"{screening_score}/10"
        )


    with result_col3:

        st.metric(
            "Model Output",
            "YES" if prediction == 1 else "NO"
        )


    st.write(
        result_text
    )


    st.info(
        "The model confidence represents the calibrated machine "
        "learning output for the YES class. It should not be "
        "interpreted as a clinical probability or diagnosis."
    )


    # ========================================================
    # PROBABILITY BREAKDOWN
    # ========================================================

    st.subheader(
        "Prediction Probabilities"
    )


    probability_col1, probability_col2 = st.columns(2)


    with probability_col1:

        st.write(
            f"**NO — {probability_no:.2%}**"
        )

        st.progress(
            float(probability_no)
        )


    with probability_col2:

        st.write(
            f"**YES — {probability_yes:.2%}**"
        )

        st.progress(
            float(probability_yes)
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.header(
        "4. Model Explanation"
    )


    st.write(
        "SHAP shows which features had the greatest influence "
        "on this individual prediction."
    )


    preprocessor = (
        base_model
        .named_steps["preprocessor"]
    )


    xgb_model = (
        base_model
        .named_steps["model"]
    )


    transformed_input = (
        preprocessor.transform(
            input_data
        )
    )


    explainer = shap.TreeExplainer(
        xgb_model
    )


    shap_values = explainer.shap_values(
        transformed_input
    )


    if len(shap_values.shape) == 2:

        shap_values = shap_values[0]


    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    explanation = pd.DataFrame(
        {
            "Feature": feature_names,

            "Impact": shap_values
        }
    )


    explanation["Absolute Impact"] = (
        explanation["Impact"].abs()
    )


    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )


    top_features = (
        explanation
        .head(8)
        .copy()
    )


    def clean_feature_name(name):

        name = name.replace(
            "numeric__",
            ""
        )

        name = name.replace(
            "categorical__",
            ""
        )

        return name


    top_features["Feature"] = (
        top_features["Feature"]
        .apply(clean_feature_name)
    )


    top_features = (
        top_features
        .sort_values("Impact")
    )


    # --------------------------------------------------------
    # SHAP PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )


    ax.barh(
        top_features["Feature"],
        top_features["Impact"]
    )


    ax.axvline(
        0,
        linewidth=0.8
    )


    ax.set_xlabel(
        "SHAP Impact",
        fontsize=13
    )


    ax.set_ylabel(
        "",
        fontsize=13
    )


    ax.set_title(
        "Top Features Influencing the Prediction",
        fontsize=16,
        fontweight="bold"
    )


    plt.tight_layout()


    st.pyplot(
        fig,
        use_container_width=True
    )


    plt.close(fig)


    st.caption(
        "SHAP explains how the model used the input features. "
        "It does not establish medical causation."
    )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    st.divider()


    with st.expander(
        "Technical Details"
    ):

        st.write(
            "**Model:** Calibrated XGBoost classifier"
        )

        st.write(
            "**Explainability:** SHAP TreeExplainer"
        )

        st.write(
            "**Training records:** 704"
        )

        st.write(
            "**Input features:** 17"
        )

        st.write(
            "**Evaluation accuracy:** 99.29%"
        )

        st.write(
            "**Evaluation F1 score:** 98.67%"
        )

        st.write(
            "**Supported age range:** 17–64"
        )

        st.write(
            "**Excluded feature:** Family-history autism variable"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ASD Screening Tool<br>
        Python · XGBoost · Scikit-learn · SHAP · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)