
import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

DATA_PATH = DATA_DIR / "ecommerce_cleaned.csv"
SEGMENT_PATH = DATA_DIR / "customer_segments.csv"
FORECAST_PATH = DATA_DIR / "demand_forecast_next_3_months.csv"
INVENTORY_PATH = DATA_DIR / "inventory_recommendations.csv"
CHURN_REPORT_PATH = REPORT_DIR / "churn_customer_report.txt"

st.set_page_config(
    page_title="RetailPulse | Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* Existing CSS code inga irukkum */

/* INGA add pannu */
.stDownloadButton button {
    background-color: #2563eb !important;
    color: white !important;
    font-weight: 700 !important;
}

.stDownloadButton button p,
.stDownloadButton button span {
    color: white !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    if "Purchase Date" in df.columns:
        df["Purchase Date"] = pd.to_datetime(
            df["Purchase Date"], errors="coerce"
        )

    return df


def load_csv(path):
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


def show_missing_file(path, command):
    st.warning(f"Required output not found: {path.name}")
    st.code(command, language="powershell")


def show_header(title, subtitle):
    st.markdown(f"""
    <div style="
        background: linear-gradient(110deg, #ffffff, #edf5ff);
        border: 1px solid #dbeafe;
        border-left: 5px solid #2878c8;
        border-radius: 13px;
        padding: 17px 22px;
        margin-bottom: 20px;
    ">
        <div style="
            color: #12315b;
            font-size: 25px;
            font-weight: 800;
        ">{title}</div>
        <div style="
            color: #64748b;
            font-size: 13px;
            margin-top: 5px;
        ">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MAIN DATA
# --------------------------------------------------
st.markdown("""
<div class="rp-header">
    <div class="rp-eyebrow">AI-POWERED RETAIL INTELLIGENCE</div>
    <div class="rp-title">📊 RetailPulse</div>
    <div class="rp-subtitle">
        Customer Analytics · Demand Forecasting · Churn Insights · Inventory Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load cleaned dataset: {exc}")
    st.info("Check that data/processed/ecommerce_cleaned.csv exists.")
    st.stop()

required_columns = [
    "Customer ID",
    "Purchase Date",
    "Product Category",
    "Product Price",
    "Quantity",
    "Total Purchase Amount",
    "Payment Method",
    "Churn"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(
        "Dataset is missing columns: " + ", ".join(missing_columns)
    )
    st.stop()

# Numeric conversions for consistent calculations
for col in [
    "Product Price",
    "Quantity",
    "Total Purchase Amount",
    "Churn"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 6px 0 18px;">
        <div style="font-size: 23px; font-weight: 850; color: #12315b;">
            📊 RetailPulse
        </div>
        <div style="
            font-size: 11px;
            color: #2878c8;
            font-weight: 750;
            letter-spacing: 1.5px;
            margin-top: 4px;
        ">ANALYTICS WORKSPACE</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("#### Navigation")

    page = st.radio(
        "Select Module",
        [
            "Sales Analysis",
            "Customer Segments",
            "Demand Forecast",
            "Churn Analysis",
            "Inventory Insights"
        ],
        format_func=lambda x: {
            "Sales Analysis": "📈  Sales Analysis",
            "Customer Segments": "👥  Customer Segments",
            "Demand Forecast": "🔮  Demand Forecast",
            "Churn Analysis": "🔍  Churn Analysis",
            "Inventory Insights": "📦  Inventory Insights"
        }[x],
        label_visibility="collapsed",
        key="retailpulse_navigation"
    )

    st.divider()

    st.markdown("#### Dataset Overview")
    st.caption(f"📁 Records: {len(df):,}")
    st.caption(f"👥 Unique customers: {df['Customer ID'].nunique():,}")
    st.caption(f"🛍️ Categories: {df['Product Category'].nunique():,}")

    if "Purchase Date" in df.columns:
        valid_dates = df["Purchase Date"].dropna()
        if not valid_dates.empty:
            st.caption(
                "📅 Data period: "
                f"{valid_dates.min():%Y-%m} to {valid_dates.max():%Y-%m}"
            )


# --------------------------------------------------
# 1. SALES ANALYSIS
# --------------------------------------------------
if page == "Sales Analysis":
    show_header(
        "📈 Sales Analysis",
        "Monitor sales performance, category contribution and transaction trends."
    )

    total_sales = df["Total Purchase Amount"].sum()
    transactions = len(df)
    average_purchase = df["Total Purchase Amount"].mean()
    categories = df["Product Category"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"{total_sales:,.0f}")
    c2.metric("Transactions", f"{transactions:,}")
    c3.metric("Average Purchase", f"{average_purchase:,.2f}")
    c4.metric("Product Categories", f"{categories:,}")

    st.markdown("### Performance Overview")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Sales by Product Category")

        category_sales = (
            df.groupby("Product Category")["Total Purchase Amount"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(category_sales, color="#2878c8")

    with right:
        st.markdown("#### Payment Method Distribution")

        payment_counts = (
            df["Payment Method"]
            .fillna("Unknown")
            .value_counts()
        )

        st.bar_chart(payment_counts, color="#60a5fa")

    st.markdown("### Monthly Sales Trend")

    monthly_sales = (
        df.dropna(subset=["Purchase Date"])
        .set_index("Purchase Date")
        .resample("ME")["Total Purchase Amount"]
        .sum()
    )

    st.line_chart(monthly_sales, color="#1769aa")

    st.markdown("### Category Sales Details")

    category_table = (
        category_sales.rename("Total Sales")
        .reset_index()
        .sort_values("Total Sales", ascending=False)
    )

    st.dataframe(
        category_table,
        width="stretch",
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Sales Summary",
        data=category_table.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_sales_summary.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 2. CUSTOMER SEGMENTS
# --------------------------------------------------
elif page == "Customer Segments":
    show_header(
        "👥 Customer Segments",
        "Explore customer groups created by the configured segmentation model."
    )

    segments = load_csv(SEGMENT_PATH)

    if segments is None:
        show_missing_file(
            SEGMENT_PATH,
            "python -u .\\src\\customer_segmentation.py"
        )
    else:
        segment_column = next(
            (
                c for c in [
                    "Segment",
                    "Customer_Segment",
                    "Customer Segment",
                    "Cluster"
                ]
                if c in segments.columns
            ),
            None
        )

        c1, c2 = st.columns(2)
        c1.metric("Customers in Output", f"{len(segments):,}")

        if segment_column:
            segment_counts = (
                segments[segment_column]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
            )

            c2.metric("Segment Groups", f"{len(segment_counts):,}")

            st.markdown("### Customer Distribution")

            left, right = st.columns([1, 1.5])

            with left:
                st.markdown("#### Customers by Segment")
                st.bar_chart(segment_counts, color="#2878c8")

            with right:
                st.markdown("#### Segment Summary")

                segment_table = (
                    segment_counts.rename("Customer Count")
                    .rename_axis(segment_column)
                    .reset_index()
                )

                st.dataframe(
                    segment_table,
                    width="stretch",
                    hide_index=True
                )

        else:
            st.info(
                "No recognized segment column was found. "
                "Showing the available output columns."
            )

        st.markdown("### Segmentation Results")

        st.dataframe(
            segments.head(500),
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Customer Segments",
            data=segments.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_customer_segments.csv",
            mime="text/csv"
        )

        st.caption(
            "Segment labels and interpretation depend on the features "
            "and naming rules in the segmentation script."
        )


# --------------------------------------------------
# 3. DEMAND FORECAST
# --------------------------------------------------
elif page == "Demand Forecast":
    show_header(
        "🔮 Demand Forecast",
        "Review the saved model forecast alongside historical demand."
    )

    forecast = load_csv(FORECAST_PATH)

    if forecast is None:
        show_missing_file(
            FORECAST_PATH,
            "python -u .\\src\\demand_forecasting.py"
        )
    else:
        needed = {"Month", "Product Category", "Forecast_Demand"}

        if not needed.issubset(forecast.columns):
            st.error(
                "Forecast file has unexpected columns. Available columns: "
                + ", ".join(forecast.columns)
            )
        else:
            forecast["Month"] = pd.to_datetime(
                forecast["Month"], errors="coerce"
            )

            forecast["Forecast_Demand"] = pd.to_numeric(
                forecast["Forecast_Demand"], errors="coerce"
            )

            forecast = forecast.dropna(
                subset=["Month", "Product Category", "Forecast_Demand"]
            )

            st.warning(
                "Historical demonstration only: the available training "
                "data ends in September 2023. The saved forecast covers "
                "October–December 2023, not current 2026 demand."
            )

            if not forecast.empty:
                c1, c2, c3 = st.columns(3)

                c1.metric("Forecast Records", f"{len(forecast):,}")
                c2.metric(
                    "Categories",
                    f"{forecast['Product Category'].nunique():,}"
                )
                c3.metric(
                    "Total Forecast Demand",
                    f"{forecast['Forecast_Demand'].sum():,.0f}"
                )

                st.markdown("### Forecast by Month and Category")

                chart_data = forecast.pivot_table(
                    index="Month",
                    columns="Product Category",
                    values="Forecast_Demand",
                    aggfunc="sum"
                ).sort_index()

                st.line_chart(chart_data)

                st.markdown("### Forecast Results")

                display_forecast = forecast.copy()
                display_forecast["Month"] = (
                    display_forecast["Month"].dt.strftime("%Y-%m")
                )

                st.dataframe(
                    display_forecast.sort_values(
                        ["Month", "Product Category"]
                    ),
                    width="stretch",
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Forecast CSV",
                    data=forecast.to_csv(index=False).encode("utf-8"),
                    file_name="retailpulse_demand_forecast.csv",
                    mime="text/csv"
                )

    st.markdown("### Historical Daily Quantity")

    daily = (
        df.dropna(subset=["Purchase Date"])
        .set_index("Purchase Date")
        .resample("D")["Quantity"]
        .sum()
    )

    st.line_chart(daily, color="#2878c8")

    st.dataframe(
        daily.tail(30).rename("Quantity").reset_index(),
        width="stretch",
        hide_index=True
    )


# --------------------------------------------------
# 4. CHURN ANALYSIS
# --------------------------------------------------
elif page == "Churn Analysis":
    show_header(
        "🔍 Customer Churn Analysis",
        "Inspect existing churn labels and review model evaluation limitations."
    )

    churn_values = pd.to_numeric(df["Churn"], errors="coerce")
    valid_churn = churn_values.dropna()

    churned_count = int((valid_churn == 1).sum())
    label_rate = (
        (valid_churn == 1).mean() * 100
        if not valid_churn.empty else 0
    )

    unique_customers = df["Customer ID"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Transaction Records", f"{len(df):,}")
    c2.metric("Records Labelled Churned", f"{churned_count:,}")
    c3.metric("Record-Level Churn Label Rate", f"{label_rate:.2f}%")

    st.markdown("### Churn Label Distribution")

    churn_counts = valid_churn.value_counts().sort_index()
    churn_counts.index = [
        f"Label {int(value)}" for value in churn_counts.index
    ]

    st.bar_chart(churn_counts, color="#2878c8")

    st.warning(
        "The current customer-level churn model has ROC-AUC approximately "
        "0.50, which is close to random ranking. Do not use its predictions "
        "for business decisions until the churn label, features and "
        "evaluation approach are validated and the model is improved."
    )

    st.info(
        "These figures describe existing dataset labels, not confirmed "
        "future customer churn. Transaction-level label counts are not "
        "the same as unique-customer counts."
    )

    if CHURN_REPORT_PATH.exists():
        with st.expander("📄 View Customer Churn Model Report"):
            try:
                st.text(
                    CHURN_REPORT_PATH.read_text(encoding="utf-8")
                )
            except Exception as exc:
                st.error(f"Could not read churn report: {exc}")
    else:
        st.info("Customer churn model report was not found.")


# --------------------------------------------------
# 5. INVENTORY INSIGHTS
# --------------------------------------------------
elif page == "Inventory Insights":
    show_header(
        "📦 Inventory Insights",
        "Review category-level reorder recommendations based on saved demand estimates."
    )

    inventory = load_csv(INVENTORY_PATH)

    if inventory is None:
        show_missing_file(
            INVENTORY_PATH,
            "python -u .\\src\\inventory_recommendation.py"
        )
    else:
        expected = {
            "Product Category",
            "Current_Stock",
            "Recommended_Stock_Level",
            "Reorder_Quantity",
            "Recommendation"
        }

        if not expected.issubset(inventory.columns):
            st.error(
                "Inventory file has unexpected columns. Available columns: "
                + ", ".join(inventory.columns)
            )
        else:
            inventory["Reorder_Quantity"] = pd.to_numeric(
                inventory["Reorder_Quantity"], errors="coerce"
            ).fillna(0)

            inventory["Current_Stock"] = pd.to_numeric(
                inventory["Current_Stock"], errors="coerce"
            )

            inventory["Recommended_Stock_Level"] = pd.to_numeric(
                inventory["Recommended_Stock_Level"], errors="coerce"
            )

            reorder_total = inventory["Reorder_Quantity"].sum()
            reorder_categories = int(
                (inventory["Reorder_Quantity"] > 0).sum()
            )

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Product Categories",
                f"{inventory['Product Category'].nunique():,}"
            )
            c2.metric("Total Reorder Quantity", f"{reorder_total:,.0f}")
            c3.metric("Categories to Reorder", f"{reorder_categories:,}")

            st.markdown("### Reorder Quantity by Category")

            chart_inventory = inventory.set_index("Product Category")[
                "Reorder_Quantity"
            ]

            st.bar_chart(chart_inventory, color="#2878c8")

            st.markdown("### Inventory Recommendation Table")

            st.dataframe(
                inventory,
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Inventory Recommendations",
                data=inventory.to_csv(index=False).encode("utf-8"),
                file_name="retailpulse_inventory_recommendations.csv",
                mime="text/csv"
            )

            st.info(
                "Recommendations use the saved stock inputs and the "
                "assumptions configured in the inventory script. "
                "Verify current stock, lead time and safety stock before "
                "using recommendations operationally."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.markdown("""
<div class="rp-footer">
    RetailPulse Analytics · Customer Insights · Demand Planning
</div>
""", unsafe_allow_html=True)