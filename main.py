# main.py

from src.data_load import load_ordr, load_rdr1
from src.model import predict_revenue_sm, predict_revenue_lasso, predict_revenue_ridge
from src.evaluate import evaluate_sales_revenue_regression

import pandas as pd
import os

def main():
    # Step 1: Load data
    print("Loading data...")
    df_ordr = load_ordr()
    df_rdr1 = load_rdr1()

    df = pd.merge(df_ordr, df_rdr1, on="DocEntry", how="inner")
    print(f"Merged dataset contains {len(df)} rows.")

    # Step 2: Run forecasting models
    print("Running revenue prediction models...")
    df_ols = predict_revenue_sm(df)
    df_lasso = predict_revenue_lasso(df)
    df_ridge = predict_revenue_ridge(df)

    # Step 3: Visualization and evaluation
    print("Evaluating and plotting model results...")
    evaluate_sales_revenue_regression(df)

    # Step 4: Save outputs
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    df_ols.to_csv(os.path.join(output_dir, "ols_forecast.csv"), index=False)
    df_lasso.to_csv(os.path.join(output_dir, "lasso_forecast.csv"), index=False)
    df_ridge.to_csv(os.path.join(output_dir, "ridge_forecast.csv"), index=False)

    print("All forecasts saved to the /output directory.")

if __name__ == "__main__":
    main()
