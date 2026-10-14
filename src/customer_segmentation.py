
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

# Project folders
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "ecommerce_cleaned.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned data
df = pd.read_csv(DATA_FILE)

required = [
    "Customer ID",
    "Total Purchase Amount",
    "Quantity",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Build one row per customer
df["Total Purchase Amount"] = pd.to_numeric(
    df["Total Purchase Amount"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

customer_data = (
    df.dropna(subset=["Customer ID"])
    .groupby("Customer ID")
    .agg(
        Total_Spending=("Total Purchase Amount", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Purchase_Count=("Customer ID", "size"),
    )
    .reset_index()
)

features = [
    "Total_Spending",
    "Total_Quantity",
    "Purchase_Count",
]

customer_data[features] = customer_data[features].fillna(0)
customer_data[features] = customer_data[features].clip(lower=0)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(customer_data[features])

# Train K-Means model
if len(customer_data) < 4:
    raise ValueError("At least 4 customers are required for 4 clusters.")

model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
)
customer_data["Cluster"] = model.fit_predict(X)

# Evaluate clustering quality when possible
if len(customer_data) > 4 and customer_data["Cluster"].nunique() > 1:
    score = silhouette_score(X, customer_data["Cluster"])
    print(f"Silhouette Score: {score:.3f}")

# Give clusters meaningful names based on spending rank
cluster_spending = customer_data.groupby("Cluster")["Total_Spending"].mean()
ranked_clusters = cluster_spending.sort_values(ascending=False).index.tolist()

segment_names = [
    "High Value",
    "Regular",
    "Occasional",
    "Low Spend",
]

name_map = {
    cluster: segment_names[rank]
    for rank, cluster in enumerate(ranked_clusters)
}

customer_data["Customer_Segment"] = customer_data["Cluster"].map(name_map)

# Save results and trained objects
output_file = OUTPUT_DIR / "customer_segments.csv"
customer_data.to_csv(output_file, index=False)

joblib.dump(model, MODEL_DIR / "customer_segmentation_model.pkl")
joblib.dump(scaler, MODEL_DIR / "customer_segmentation_scaler.pkl")

print("\nCustomer Segmentation Completed!")
print(f"Total customers: {len(customer_data):,}")
print("\nSegment counts:")
print(customer_data["Customer_Segment"].value_counts())
print(f"\nResults saved to: {output_file}")
print(f"Model saved to: {MODEL_DIR}")