
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Project paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

INPUT_FILE = RAW_DIR / "ecommerce_customer_data_custom_ratios.csv"
OUTPUT_FILE = PROCESSED_DIR / "ecommerce_cleaned.csv"
REPORT_FILE = REPORTS_DIR / "data_quality_report.json"


def clean_column_names(df):
    """Standardize column names without changing their meaning."""
    df.columns = [
        str(col).strip().replace("\n", " ")
        for col in df.columns
    ]
    return df


def main():
    # Create output folders if they do not exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Place the CSV file inside data/raw/ and check its name."
        )

    # Read original CSV; do not modify the source file
    df = pd.read_csv(INPUT_FILE)
    initial_rows = len(df)

    df = clean_column_names(df)

    # Convert blank strings and whitespace-only cells to missing values
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Remove exact duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Parse dates safely
    if "Purchase Date" in df.columns:
        df["Purchase Date"] = pd.to_datetime(
            df["Purchase Date"], errors="coerce"
        )

    # Convert known numeric columns safely
    numeric_columns = [
        "Product Price",
        "Quantity",
        "Total Purchase Amount",
        "Customer Age",
        "Age",
        "Returns",
        "Churn",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove redundant Age column only if it exactly matches Customer Age
    if "Age" in df.columns and "Customer Age" in df.columns:
        both_present = df["Age"].notna() & df["Customer Age"].notna()
        if (
            both_present.any()
            and df.loc[both_present, "Age"].equals(
                df.loc[both_present, "Customer Age"]
            )
        ):
            df = df.drop(columns=["Age"])

    # Standardize text fields without changing category labels
    text_columns = [
        "Product Category",
        "Payment Method",
        "Gender",
        "Customer Name",
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # Check suspicious values; report them instead of silently deleting
    issues = {}

    if "Product Price" in df.columns:
        issues["nonpositive_product_price"] = int(
            (df["Product Price"] <= 0).fillna(False).sum()
        )

    if "Quantity" in df.columns:
        issues["nonpositive_quantity"] = int(
            (df["Quantity"] <= 0).fillna(False).sum()
        )

    if "Total Purchase Amount" in df.columns:
        issues["negative_purchase_amount"] = int(
            (df["Total Purchase Amount"] < 0).fillna(False).sum()
        )

    if "Returns" in df.columns:
        valid_returns = df["Returns"].dropna().isin([0, 1])
        issues["returns_values_outside_0_1"] = int(
            len(df["Returns"].dropna()) - valid_returns.sum()
        )

    if "Churn" in df.columns:
        valid_churn = df["Churn"].dropna().isin([0, 1])
        issues["churn_values_outside_0_1"] = int(
            len(df["Churn"].dropna()) - valid_churn.sum()
        )

    # Do not guess missing Returns or Churn labels.
    # Keep missing values so later modelling can handle them explicitly.

    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d %H:%M:%S")

    # Build quality report
    report = {
        "input_file": INPUT_FILE.name,
        "rows_before_cleaning": initial_rows,
        "rows_after_cleaning": len(df),
        "exact_duplicate_rows_removed": duplicate_rows,
        "columns_after_cleaning": list(df.columns),
        "missing_values_by_column": {
            str(k): int(v)
            for k, v in df.isna().sum().items()
        },
        "suspicious_value_counts": issues,
        "date_min": (
            str(df["Purchase Date"].min())
            if "Purchase Date" in df.columns
            else None
        ),
        "date_max": (
            str(df["Purchase Date"].max())
            if "Purchase Date" in df.columns
            else None
        ),
        "notes": [
            "Missing Returns values were not automatically replaced with zero.",
            "Missing Churn labels were not automatically imputed.",
            "Suspicious numeric values are reported, not silently deleted.",
            "Original source CSV was not modified.",
        ],
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDATA CLEANING COMPLETED")
    print("-" * 45)
    print(f"Rows before cleaning : {initial_rows:,}")
    print(f"Rows after cleaning  : {len(df):,}")
    print(f"Duplicates removed   : {duplicate_rows:,}")
    print(f"Columns               : {len(df.columns)}")
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print("\nSaved cleaned data:")
    print(OUTPUT_FILE)
    print("\nSaved quality report:")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()