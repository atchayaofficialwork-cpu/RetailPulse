
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "ecommerce_cleaned.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...", flush=True)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

target = "Churn"

if target not in df.columns:
    raise ValueError("Dataset must contain a 'Churn' column.")

df = df.dropna(subset=[target]).copy()
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])
df = df[df[target].isin([0, 1])].copy()
df[target] = df[target].astype(int)

if df[target].nunique() != 2:
    raise ValueError("Churn must contain both 0 and 1.")

# Use Customer ID for splitting only, never as a model feature.
if "Customer ID" in df.columns:
    groups = df["Customer ID"].fillna("__MISSING_ID__").astype(str)
else:
    groups = None

# Exclude identifiers and personal information.
drop_columns = [
    "Customer ID",
    "Customer Name",
    target,
]

# Conservative exclusions: without timestamps proving these values
# are known before churn, they could introduce leakage.
possible_leakage = [
    "Returns",
    "Total Purchase Amount",
    "TotalPurchase Amount",
]

drop_columns += [
    col for col in possible_leakage if col in df.columns
]

X = df.drop(columns=drop_columns, errors="ignore").copy()
y = df[target].copy()

# Convert purchase date to numeric features.
if "Purchase Date" in X.columns:
    dates = pd.to_datetime(X["Purchase Date"], errors="coerce")
    X["Purchase_Year"] = dates.dt.year
    X["Purchase_Month"] = dates.dt.month
    X["Purchase_DayOfWeek"] = dates.dt.dayofweek
    X = X.drop(columns=["Purchase Date"])

# Convert known numeric fields safely.
numeric_candidates = [
    "Product Price",
    "Quantity",
    "Customer Age",
]

for col in numeric_candidates:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

# Remove columns that contain no usable information.
X = X.dropna(axis=1, how="all")

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()

if not numeric_features and not categorical_features:
    raise ValueError("No usable features found in the dataset.")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

print("Splitting data...", flush=True)

# Keep each customer's records entirely in either train or test.
if groups is not None and groups.nunique() >= 2:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "Train/test split does not contain both churn classes. "
        "Check the dataset and customer grouping."
    )

print(f"Training rows: {len(X_train):,}", flush=True)
print(f"Testing rows: {len(X_test):,}", flush=True)
print("Training model...", flush=True)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)
pr_auc = average_precision_score(y_test, probabilities)

report = classification_report(
    y_test,
    predictions,
    zero_division=0,
)
matrix = confusion_matrix(y_test, predictions)

print("\nCHURN MODEL EVALUATION")
print("=" * 30)
print(f"Rows used: {len(df):,}")
print(f"Features used: {X.shape[1]}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(matrix)

model_path = MODEL_DIR / "churn_prediction_model.pkl"
joblib.dump(model, model_path)

report_path = REPORT_DIR / "churn_model_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("CHURN PREDICTION MODEL REPORT\n")
    file.write("=" * 35 + "\n")
    file.write(f"Rows used: {len(df)}\n")
    file.write(f"Features used: {X.shape[1]}\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n")
    file.write(f"ROC-AUC: {auc:.4f}\n")
    file.write(f"PR-AUC: {pr_auc:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(report + "\n")
    file.write("Confusion Matrix:\n")
    file.write(str(matrix) + "\n")
    file.write("\nExcluded possible leakage columns when present:\n")
    file.write(", ".join(
        col for col in possible_leakage if col in df.columns
    ) + "\n")

print(f"\nModel saved: {model_path}")
print(f"Report saved: {report_path}")
print("\nChurn model training completed.")