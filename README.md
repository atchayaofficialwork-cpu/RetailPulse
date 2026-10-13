# RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform

## Project Overview

RetailPulse is a data analytics and machine learning platform designed to
analyze retail sales, segment customers, forecast product demand, estimate
customer churn, and generate inventory recommendations.

## Features

1. Sales Analysis – Explore sales performance and product categories.
2. Customer Segmentation – Group customers based on purchasing behaviour.
3. Demand Forecast – Forecast demand by product category.
4. Churn Analysis – Evaluate customer churn patterns and model predictions.
5. Inventory Insights – Recommend stock levels and reorder quantities.

## Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Project Structure

    RetailPulse/
    ├── app/
    │   └── dashboard.py
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── models/
    ├── reports/
    ├── src/
    │   ├── eda.py
    │   ├── customer_segmentation.py
    │   ├── demand_forecasting.py
    │   ├── churn_prediction.py
    │   ├── churn_customer_model.py
    │   └── inventory_recommendation.py
    ├── requirements.txt
    └── README.md

## Installation

1. Install Python.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment.

Windows PowerShell:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

4. Install dependencies:

    python -m pip install -r requirements.txt

## Run the Dashboard

    python -m streamlit run app/dashboard.py

Streamlit will display a local URL in the terminal. Open that URL in
your browser to use the dashboard.

## Data and Outputs

- data/raw/ contains the original dataset.
- data/processed/ contains cleaned data and generated analytics.
- models/ contains trained machine learning models.
- reports/ contains evaluation reports and charts.

## Important Limitations

- Churn model evaluation currently shows ROC-AUC close to 0.50.
  Its predictions should not be treated as reliable until the model
  and dataset labels are investigated further.
- The available dataset contains transactions through September 2023.
  Forecasts generated from it do not represent verified current demand.
- Inventory recommendations depend on assumptions about lead time,
  safety stock, and current inventory.
- Results should be validated with current business data before
  making real purchasing or operational decisions.

## Disclaimer

This project is intended for educational and analytical purposes.
Model outputs are estimates and should be validated before business use.
