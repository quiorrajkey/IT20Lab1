# main.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

MODEL_PATH = "models/wine_quality_pipeline.pkl"
DEFAULT_CSV_PATH = "dataset/Redwinequality.csv"  # your file from VSCode tree


# ---- helper functions -----------------------------------------------------
def normalize_colnames(df):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df


EXPECTED_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",   # may be "free_sulfur_dioxide" or "free_sulfur_dioxide"
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]


@st.cache_data
def load_data_from_path(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    df = normalize_colnames(df)
    return df


def ensure_features_exist(df):
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    return missing


def prepare_binary_target(df):
    if "quality" not in df.columns:
        raise ValueError("The file must contain a 'quality' column (numeric).")
    df = df.copy()
    df["good_quality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)
    return df


def build_and_train_pipeline(X_train, y_train, n_estimators=200, random_state=42):
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_save_model(df, save_path=MODEL_PATH):
    df = prepare_binary_target(df)
    X = df[EXPECTED_FEATURES]
    y = df["good_quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_and_train_pipeline(X_train, y_train)

    # evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    # cross-val
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

    # save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(pipeline, save_path)

    return {
        "pipeline": pipeline,
        "accuracy": acc,
        "report": report,
        "cv_scores": cv_scores,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ---- UI -------------------------------------------------------------------
st.title("🍷 Wine Quality — Good vs Not Good (Binary)")

st.markdown(
    """
**Definition:** `good_quality = 1` if `quality >= 7`, otherwise `0`.  
This app can train a model from your CSV and let you enter new samples to get a prediction + confidence.
"""
)

# Left column: dataset + training controls
left, right = st.columns([1, 1])

with left:
    st.header("1) Load dataset")
    uploaded_file = st.file_uploader("Upload a CSV (optional). If none, app will try default dataset path.", type=["csv"])

    if uploaded_file is not None:
        df = load_data_from_path(uploaded_file)
        st.success("Dataset uploaded.")
    else:
        if os.path.exists(DEFAULT_CSV_PATH):
            df = load_data_from_path(DEFAULT_CSV_PATH)
            st.info(f"Loaded default file: `{DEFAULT_CSV_PATH}`")
        else:
            st.warning("No dataset uploaded and default file not found. Upload a CSV to continue.")
            df = None

    if df is not None:
        st.subheader("Preview & diagnostics")
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head(10))
        st.write("Missing values per column:")
        st.table(df.isna().sum())

        # check expected features
        missing = ensure_features_exist(df)
        if missing:
            st.error(f"Missing expected feature columns (rename in the CSV or adjust): {missing}")
        else:
            st.success("All expected chemical columns present.")

        if "quality" not in df.columns:
            st.error("`quality` column is required to train. If you only want to predict, you can still load a saved model.")
        else:
            st.info("Quality column present (we will convert to binary 'good_quality' for training).")

        st.markdown("---")
        st.header("2) Train model (optional)")
        st.write(
            "If you press **Train model**, the app will train a RandomForest pipeline (median imputation + scaling) "
            "and save it to `models/wine_quality_pipeline.pkl`."
        )

        if st.button("🚀 Train model"):
            with st.spinner("Training model — this may take ~10–30 seconds"):
                try:
                    results = train_save_model(df)
                    st.success(f"Model trained and saved to `{MODEL_PATH}`.")
                    st.write(f"Test accuracy: **{results['accuracy']:.3f}**")
                    st.write("Cross-val accuracy (5 folds):")
                    st.write(results["cv_scores"])
                    st.text("Classification report (test set):")
                    st.text(results["report"])

                    # show feature importances
                    clf = results["pipeline"].named_steps["clf"]
                    importances = clf.feature_importances_
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(EXPECTED_FEATURES[::-1], importances[::-1])
                    ax.set_title("Feature importances (RandomForest)")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Training failed: {e}")

        st.markdown("---")
        st.header("3) Load or inspect saved model")
        if os.path.exists(MODEL_PATH):
            st.success(f"Saved model found at `{MODEL_PATH}`")
            if st.button("🔁 Load saved model"):
                pipeline = load_model(MODEL_PATH)
                st.session_state["pipeline_loaded"] = True
                st.session_state["pipeline"] = pipeline
                st.success("Model loaded into session.")
        else:
            st.info("No saved model found. Train a model to create one.")

with right:
    st.header("4) Predict a new sample")
    # Model availability
    pipeline = st.session_state.get("pipeline") if st.session_state.get("pipeline") is not None else load_model()

    # If we have a dataset, compute medians for defaults
    defaults = {k: 0.0 for k in EXPECTED_FEATURES}
    if df is not None:
        # safe: only use present features
        present = [c for c in EXPECTED_FEATURES if c in df.columns]
        if present:
            med = df[present].median()
            for k in med.index:
                defaults[k] = float(med[k])

    st.write("Enter chemical measurements (values pre-filled with dataset medians if dataset provided).")

    user_inputs = {}
    for feat in EXPECTED_FEATURES:
        # set ranges and step heuristically
        step = 0.01 if feat not in ["free_sulfur_dioxide", "total_sulfur_dioxide", "alcohol"] else 0.1
        fmt = float
        user_inputs[feat] = st.number_input(feat.replace("_", " ").title(), value=defaults.get(feat, 0.0), step=step, format="%.4f")

    st.markdown("**Prediction action**")
    if pipeline is None:
        st.warning("No model loaded. Either train a model above or upload a pretrained `models/wine_quality_pipeline.pkl`.")
    else:
        if st.button("Predict quality"):
            sample = np.array([user_inputs[feat] for feat in EXPECTED_FEATURES]).reshape(1, -1)
            try:
                proba = pipeline.predict_proba(sample)[0]  # [prob_not_good, prob_good]
                pred = pipeline.predict(sample)[0]
                conf = proba[pred]

                if pred == 1:
                    st.success(f"✅ Predicted: **Good quality** (>=7). Confidence: **{conf:.2f}**")
                else:
                    st.error(f"❌ Predicted: **Not good** (<7). Confidence: **{conf:.2f}**")

                st.write("Probability breakdown:")
                st.write({"not_good (0)": float(proba[0]), "good (1)": float(proba[1])})

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.header("Notes / Next steps")
    st.write(
        """
- If you want more calibration of probabilities, we can add probability calibration (e.g., `CalibratedClassifierCV`).  
- To deploy publicly: push repo to GitHub and connect to Streamlit Cloud (instructions below).
- I can also produce a separate Jupyter notebook for training and exporting the model (if you prefer).
"""
    )
