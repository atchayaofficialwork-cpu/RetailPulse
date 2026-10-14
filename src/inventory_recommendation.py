
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FORECAST_FILE = (
    ROOT / "data" / "processed"
    / "demand_forecast_next_3_months.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading demand forecasts...", flush=True)

forecast = pd.read_csv(FORECAST_FILE)
forecast["Month"] = pd.to_datetime(forecast["Month"], errors="coerce")
forecast["Forecast_Demand"] = pd.to_numeric(
    forecast["Forecast_Demand"], errors="coerce"
)

forecast = forecast.dropna(
    subset=["Month", "Product Category", "Forecast_Demand"]
)

if forecast.empty:
    raise ValueError("No valid demand forecast records found.")

# Total forecast demand over the next three months
summary = (
    forecast.groupby("Product Category", as_index=False)
    .agg(
        Forecast_3_Month_Demand=("Forecast_Demand", "sum"),
        Average_Monthly_Demand=("Forecast_Demand", "mean"),
    )
)

# Assumptions: one-month lead time and 20% safety stock
summary["Safety_Stock"] = (
    summary["Average_Monthly_Demand"] * 0.20
).round().astype(int)

summary["Recommended_Stock_Level"] = (
    summary["Average_Monthly_Demand"] + summary["Safety_Stock"]
).round().astype(int)

# Ask for current inventory per category
print("\nEnter current stock for each category.", flush=True)
print("Use 0 if there is no stock.", flush=True)

current_stock = []

for category in summary["Product Category"]:
    while True:
        try:
            value = int(input(f"Current stock for {category}: "))
            if value < 0:
                print("Stock cannot be negative. Try again.")
                continue
            current_stock.append(value)
            break
        except ValueError:
            print("Enter a whole number, for example 250.")

summary["Current_Stock"] = current_stock

summary["Reorder_Quantity"] = (
    summary["Recommended_Stock_Level"] - summary["Current_Stock"]
).clip(lower=0)

summary["Recommendation"] = summary["Reorder_Quantity"].apply(
    lambda qty: "Reorder Now" if qty > 0 else "Stock Sufficient"
)

output_file = OUTPUT_DIR / "inventory_recommendations.csv"
summary.to_csv(output_file, index=False)

report_file = REPORT_DIR / "inventory_recommendation_report.txt"

with open(report_file, "w", encoding="utf-8") as file:
    file.write("RETAILPULSE INVENTORY RECOMMENDATION REPORT\n")
    file.write("=" * 45 + "\n\n")
    file.write(
        "Assumptions: one-month lead time and 20% safety stock.\n"
    )
    file.write(summary.to_string(index=False))
    file.write("\n")

print("\nInventory Recommendation Completed!")
print(summary.to_string(index=False))
print(f"\nCSV saved: {output_file}")
print(f"Report saved: {report_file}")