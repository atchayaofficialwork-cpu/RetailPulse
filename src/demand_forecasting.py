
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "ecommerce_cleaned.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

for folder in [MODEL_DIR, OUTPUT_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("Loading dataset...", flush=True)

df = pd.read_csv(DATA_FILE)
df["Purchase Date"] = pd.to_datetime(
    df["Purchase Date"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

df = df.dropna(subset=["Purchase Date", "Quantity", "Product Category"])
df = df[df["Quantity"] >= 0].copy()

# Aggregate transaction data to monthly category-level demand
df["Month"] = df["Purchase Date"].dt.to_period("M").dt.to_timestamp()

monthly = (
    df.groupby(["Month", "Product Category"])["Quantity"]
    .sum()
    .reset_index(name="Demand")
)

# Create a complete monthly timeline for each category
all_months = pd.date_range(
    monthly["Month"].min(),
    monthly["Month"].max(),
    freq="MS",
)

categories = sorted(monthly["Product Category"].unique())

complete_index = pd.MultiIndex.from_product(
    [all_months, categories],
    names=["Month", "Product Category"],
)

monthly = (
    monthly.set_index(["Month", "Product Category"])
    .reindex(complete_index, fill_value=0)
    .reset_index()
)

monthly = monthly.sort_values(["Product Category", "Month"])

# Time-based features and past demand features
monthly["Month_Number"] = monthly["Month"].dt.month
monthly["Year"] = monthly["Month"].dt.year
monthly["Time_Index"] = (
    (monthly["Month"].dt.year - monthly["Month"].dt.year.min()) * 12
    + monthly["Month"].dt.month
)

monthly["Lag_1"] = monthly.groupby("Product Category")["Demand"].shift(1)
monthly["Lag_2"] = monthly.groupby("Product Category")["Demand"].shift(2)
monthly["Lag_3"] = monthly.groupby("Product Category")["Demand"].shift(3)
monthly["Rolling_3_Month"] = (
    monthly.groupby("Product Category")["Demand"]
    .transform(lambda s: s.shift(1).rolling(3).mean())
)

features = [
    "Month_Number",
    "Year",
    "Time_Index",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_3_Month",
]

model_data = monthly.dropna(subset=features).copy()

# Reserve the latest 6 months for a time-based holdout test
unique_months = sorted(model_data["Month"].unique())

if len(unique_months) < 12:
    raise ValueError("At least 12 months of data are required.")

test_months = unique_months[-6:]

train = model_data[~model_data["Month"].isin(test_months)]
test = model_data[model_data["Month"].isin(test_months)]

print(f"Training rows: {len(train)}", flush=True)
print(f"Testing rows: {len(test)}", flush=True)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2,
)

print("Training demand forecasting model...", flush=True)

model.fit(train[features], train["Demand"])

predictions = model.predict(test[features]).clip(min=0)

mae = mean_absolute_error(test["Demand"], predictions)
rmse = mean_squared_error(test["Demand"], predictions) ** 0.5

print("\nDemand Forecast Evaluation")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

test_results = test[
    ["Month", "Product Category", "Demand"]
].copy()
test_results["Predicted_Demand"] = predictions
test_results.to_csv(
    OUTPUT_DIR / "demand_forecast_test_results.csv",
    index=False,
)

# Refit the final model on all available training examples
print("Refitting final model...", flush=True)
model.fit(model_data[features], model_data["Demand"])

model_path = MODEL_DIR / "demand_forecasting_model.pkl"
joblib.dump(model, model_path)

# Forecast the next 3 months using recursively updated lag features
history = monthly.copy()
future_rows = []

last_month = history["Month"].max()

for step in range(1, 4):
    forecast_month = last_month + pd.DateOffset(months=step)

    for category in categories:
        category_history = history[
            history["Product Category"] == category
        ].sort_values("Month")

        past_demand = category_history["Demand"].tolist()

        lag_1 = past_demand[-1]
        lag_2 = past_demand[-2]
        lag_3 = past_demand[-3]
        rolling_3 = sum(past_demand[-3:]) / 3

        features_row = pd.DataFrame([{
            "Month_Number": forecast_month.month,
            "Year": forecast_month.year,
            "Time_Index": (
                (forecast_month.year - monthly["Month"].dt.year.min()) * 12
                + forecast_month.month
            ),
            "Lag_1": lag_1,
            "Lag_2": lag_2,
            "Lag_3": lag_3,
            "Rolling_3_Month": rolling_3,
        }])

        forecast_value = max(
            0, float(model.predict(features_row[features])[0])
        )

        future_rows.append({
            "Month": forecast_month,
            "Product Category": category,
            "Forecast_Demand": round(forecast_value, 2),
        })

        history = pd.concat([
            history,
            pd.DataFrame([{
                "Month": forecast_month,
                "Product Category": category,
                "Demand": forecast_value,
            }]),
        ], ignore_index=True)

forecast_df = pd.DataFrame(future_rows)
forecast_file = OUTPUT_DIR / "demand_forecast_next_3_months.csv"
forecast_df.to_csv(forecast_file, index=False)

report_file = REPORT_DIR / "demand_forecast_report.txt"

with open(report_file, "w", encoding="utf-8") as f:
    f.write("DEMAND FORECASTING REPORT\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n\n")
    f.write("Next 3 months forecast:\n")
    f.write(forecast_df.to_string(index=False))

print("\nDemand Forecasting Completed!")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Model saved: {model_path}")
print(f"Test results: {OUTPUT_DIR / 'demand_forecast_test_results.csv'}")
print(f"Future forecast: {forecast_file}")
print(f"Report saved: {report_file}")