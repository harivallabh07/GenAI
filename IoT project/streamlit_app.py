import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support
)

st.set_page_config(page_title="IoT Network Traffic Analysis", layout="wide")
st.title("Cloud-Based IoT Network Traffic Monitoring")
st.caption("Dataset-based classification demo — not live network detection.")

uploaded = st.file_uploader("Upload network_traffic.csv", type=["csv"])
default_path = "network_traffic.csv"

@st.cache_data
def load_data(file_or_path):
    if hasattr(file_or_path, "read"):
        data = pd.read_csv(file_or_path)
    else:
        data = pd.read_csv(file_or_path)
    data.columns = data.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    return data

if uploaded is None:
    st.info("Upload your network_traffic.csv file to generate the dashboard.")
    st.stop()

try:
    raw_df = load_data(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

st.subheader("Dataset overview")
a, b, c = st.columns(3)
a.metric("Records", f"{len(raw_df):,}")
b.metric("Columns", f"{raw_df.shape[1]:,}")
c.metric("Missing cells", f"{int(raw_df.isna().sum().sum()):,}")
with st.expander("Preview dataset"):
    st.dataframe(raw_df.head(20), use_container_width=True)

# Find the combined label column even if its whitespace differs.
combined_candidates = [col for col in raw_df.columns if "tunnel_parents" in col and "detailed-label" in col]
if "detailed_label" not in raw_df.columns:
    if not combined_candidates:
        st.error("Could not find 'detailed_label' or the combined tunnel_parents/label/detailed-label column.")
        st.stop()
    combined_col = combined_candidates[0]
    split_labels = raw_df[combined_col].astype(str).str.split(r"\s+", n=2, expand=True)
    if split_labels.shape[1] < 3:
        st.error("The combined label column could not be split into three fields.")
        st.stop()
    raw_df[["tunnel_parents", "label", "detailed_label"]] = split_labels.iloc[:, :3].values

raw_df["detailed_label"] = raw_df["detailed_label"].astype(str).str.strip()
# In this dataset, '-' is the benign label.
raw_df["detailed_label"] = raw_df["detailed_label"].replace({"-": "Benign", "benign": "Benign"})

st.subheader("Traffic class distribution")
class_counts = raw_df["detailed_label"].value_counts()
fig, ax = plt.subplots(figsize=(9, 4))
class_counts.plot(kind="bar", ax=ax)
ax.set_title("IoT Traffic Class Distribution")
ax.set_xlabel("Traffic class")
ax.set_ylabel("Number of records")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Keep the feature preparation aligned with the notebook workflow.
df = raw_df.copy()
drop_cols = [c for c in [
    "tunnel_parents label detailed-label", "ts", "uid", "id.orig_h", "id.resp_h",
    "tunnel_parents", "label", "local_orig", "local_resp"
] if c in df.columns]
df = df.drop(columns=drop_cols, errors="ignore")

target = "detailed_label"
if target not in df.columns:
    st.error("Target column 'detailed_label' is missing after preprocessing.")
    st.stop()

# Exclude any other label-like columns to avoid target leakage.
for col in ["detailed-label", "label"]:
    if col in df.columns:
        df = df.drop(columns=[col])

y = df[target].astype(str)
X = df.drop(columns=[target])

categorical = [col for col in ["proto", "service", "conn_state", "history"] if col in X.columns]
if categorical:
    X = pd.get_dummies(X, columns=categorical, dtype=int)
X = X.replace([np.inf, -np.inf], np.nan)
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(0)
# Drop remaining nonnumeric columns if any.
X = X.select_dtypes(include=[np.number])

if y.nunique() < 2:
    st.error("At least two target classes are required to train the models.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError as exc:
    st.error(f"Could not split the dataset. Check whether every class has enough rows. Details: {exc}")
    st.stop()

with st.spinner("Training Random Forest and Logistic Regression on the uploaded dataset..."):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
lr_acc = accuracy_score(y_test, lr_pred)
st.subheader("Model performance")
m1, m2, m3 = st.columns(3)
m1.metric("Random Forest accuracy", f"{rf_acc:.2%}")
m2.metric("Logistic Regression accuracy", f"{lr_acc:.2%}")
m3.metric("Test records", f"{len(y_test):,}")

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(["Random Forest", "Logistic Regression"], [rf_acc, lr_acc])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")
for bar, val in zip(bars, [rf_acc, lr_acc]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.015, f"{val:.2%}", ha="center")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Confusion matrices")
col1, col2 = st.columns(2)
labels = sorted(y.unique())
fig1, ax1 = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, labels=labels, display_labels=labels, ax=ax1, xticks_rotation=45, colorbar=False)
ax1.set_title("Random Forest")
fig1.tight_layout()
col1.pyplot(fig1)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, lr_pred, labels=labels, display_labels=labels, ax=ax2, xticks_rotation=45, colorbar=False)
ax2.set_title("Logistic Regression")
fig2.tight_layout()
col2.pyplot(fig2)
plt.close(fig2)

st.subheader("Per-class precision, recall and F1-score")
metric_frames = []
for model_name, predictions in [("Random Forest", rf_pred), ("Logistic Regression", lr_pred)]:
    precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, labels=labels, zero_division=0)
    metric_frames.append(pd.DataFrame({
        "Class": labels, "Model": model_name, "Precision": precision,
        "Recall": recall, "F1-score": f1, "Support": support
    }))
metrics = pd.concat(metric_frames, ignore_index=True)
for metric in ["Precision", "Recall", "F1-score"]:
    pivot = metrics.pivot(index="Class", columns="Model", values=metric)
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-Class {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Traffic class")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with st.expander("Detailed classification report"):
    st.write("Random Forest")
    st.dataframe(pd.DataFrame(classification_report(y_test, rf_pred, output_dict=True, zero_division=0)).T, use_container_width=True)
    st.write("Logistic Regression")
    st.dataframe(pd.DataFrame(classification_report(y_test, lr_pred, output_dict=True, zero_division=0)).T, use_container_width=True)

st.subheader("Top Random Forest feature importances")
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(9, 5))
importance.sort_values().plot(kind="barh", ax=ax)
ax.set_title("Top 10 Feature Importances")
ax.set_xlabel("Importance")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

if "id.resp_p" in raw_df.columns:
    st.subheader("Destination port vs traffic class")
    port_table = pd.crosstab(raw_df["id.resp_p"], raw_df["detailed_label"]).head(20)
    st.dataframe(port_table, use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    port_table.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Destination Port vs Traffic Class (first 20 ports)")
    ax.set_xlabel("Destination port")
    ax.set_ylabel("Record count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Sample predictions")
sample = pd.DataFrame({
    "Actual class": y_test.reset_index(drop=True),
    "Random Forest prediction": pd.Series(rf_pred),
    "Logistic Regression prediction": pd.Series(lr_pred)
})
st.dataframe(sample.head(50), use_container_width=True)

st.warning(
    "These results are evaluated on a held-out split of the uploaded dataset. "
    "High accuracy does not by itself prove reliable real-world or live network detection."
)
