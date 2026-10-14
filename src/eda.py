import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "ecommerce_cleaned.csv"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("RETAILPULSE - EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print("\nDataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe(include="all").to_string())
print("\nMissing Values:\n", df.isnull().sum().to_string())

# 1. Sales Analysis
sales = df["Total Purchase Amount"].sum()
print(f"\nTotal Purchase Amount: {sales:,.2f}")

category_sales = (
    df.groupby("Product Category")["Total Purchase Amount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Product Category:\n", category_sales.to_string())

category_sales.plot(kind="bar", figsize=(9, 5))
plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Purchase Amount")
plt.tight_layout()
plt.savefig(REPORT_DIR / "sales_by_category.png")
plt.close()

# 2. Customer Age Analysis
df["Customer Age"].plot(kind="hist", bins=20, figsize=(8, 5))
plt.title("Customer Age Distribution")
plt.xlabel("Customer Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(REPORT_DIR / "customer_age_distribution.png")
plt.close()

# 3. Churn Analysis
print("\nChurn Distribution:\n", df["Churn"].value_counts(dropna=False).to_string())

churn_counts = df["Churn"].value_counts(dropna=False)
churn_counts.plot(kind="bar", figsize=(7, 5))
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(REPORT_DIR / "churn_distribution.png")
plt.close()

# 4. Payment Method Analysis
payment_counts = df["Payment Method"].value_counts()
print("\nPayment Method Distribution:\n", payment_counts.to_string())

payment_counts.plot(kind="bar", figsize=(8, 5))
plt.title("Payment Method Distribution")
plt.xlabel("Payment Method")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(REPORT_DIR / "payment_method_distribution.png")
plt.close()

# Save numerical summary
summary_path = REPORT_DIR / "eda_summary.txt"
with open(summary_path, "w", encoding="utf-8") as file:
    file.write(f"Dataset Shape: {df.shape}\n\n")
    file.write(f"Total Purchase Amount: {sales:,.2f}\n\n")
    file.write("Sales by Category:\n")
    file.write(category_sales.to_string())
    file.write("\n\nChurn Distribution:\n")
    file.write(churn_counts.to_string())
    file.write("\n\nMissing Values:\n")
    file.write(df.isnull().sum().to_string())

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Charts and summary saved in:", REPORT_DIR)