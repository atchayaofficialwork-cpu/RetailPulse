
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "ecommerce_cleaned.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True, parents=True)
REPORT_DIR.mkdir(exist_ok=True, parents=True)

print("Loading dataset...", flush=True)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

# Convert numeric columns
numeric_cols = [
    "Total Purchase Amount",
    "TotalPurchase Amount",
    "Product Price",
    "Quantity",
    "Returns",
    "Customer Age",
    "Churn",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# One row per customer
aggregations = {
    "Churn": ("Churn", "first"),
    "Transaction_Count": ("Customer ID", "size"),
    "Total_Quantity": ("Quantity", "sum"),
    "Average_Quantity": ("Quantity", "mean"),
    "Average_Product_Price": ("Product Price", "mean"),
    "Average_Customer_Age": ("Customer Age", "first"),
}

amount_col = (
    "Total Purchase Amount"
    if "Total Purchase Amount" in df.columns
    else "TotalPurchase Amount"
)

aggregations["Total_Spending"] = (amount_col, "sum")
aggregations["Average_Transaction_Amount"] = (amount_col, "mean")

if "Returns" in df.columns:
    aggregations["Return_Count"] = ("Returns", "sum")
    aggregations["Return_Rate"] = ("Returns", "mean")

if "Product Category" in df.columns:
    aggregations["Category_Diversity"] = (
        "Product Category",
        "nunique",
    )

customer_df = (
    df.dropna(subset=["Customer ID", "Churn"])
    .groupby("Customer ID")
    .agg(**aggregations)
    .reset_index(drop=True)
)

customer_df = customer_df.dropna(subset=["Churn"])
customer_df = customer_df[customer_df["Churn"].isin([0, 1])]
customer_df["Churn"] = customer_df["Churn"].astype(int)

X = customer_df.drop(columns=["Churn"])
y = customer_df["Churn"]

if y.nunique() != 2:
    raise ValueError("The customer-level target must contain 0 and 1.")

print(f"Unique customers: {len(customer_df):,}", flush=True)
print("Splitting customer-level data...", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=5,
        n_jobs=-1,
    )),
])

print("Training model...", flush=True)
model.fit(X_train, y_train)

print("Evaluating model...", flush=True)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, proba)
report = classification_report(y_test, pred, zero_division=0)
matrix = confusion_matrix(y_test, pred)

print("\nCustomer-level Churn Model Results")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(matrix)

model_path = MODEL_DIR / "churn_customer_model.pkl"
joblib.dump(model, model_path)

report_path = REPORT_DIR / "churn_customer_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("CUSTOMER-LEVEL CHURN MODEL\n")
    f.write(f"Customers: {len(customer_df)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC-AUC: {auc:.4f}\n\n")
    f.write(report + "\n")
    f.write(str(matrix) + "\n")

customer_df.to_csv(
    ROOT / "data" / "processed" / "customer_churn_features.csv",
    index=False,
)

print(f"\nModel saved: {model_path}")
print(f"Report saved: {report_path}")
print("Customer-level features saved.")