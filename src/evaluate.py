import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from src.model import predict_revenue_sm, predict_revenue_lasso,predict_revenue_ridge
import pandas as pd

def evaluate_sales_revenue_regression(df: pd.DataFrame):
    """
    Compare revenue predictions using OLS, Lasso, and Ridge regression,
    and plot both the actual points and the fitted lines.
    """
    df_ols = predict_revenue_sm(df)
    df_lasso = predict_revenue_lasso(df)
    df_ridge = predict_revenue_ridge(df)


    df_hist = df_ols[df_ols["Year"] != 2025]
    df_pred = df_ols[df_ols["Year"] == 2025]

    # 历史点
    x = df_hist["Year"]
    y = df_hist["rev"]

    # 预测值
    x_ols = df_pred["Year"]
    y_ols = df_pred["rev"]

    print(df_ols)
    print(df_lasso)
    print(df_ridge)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o', label="Actual Revenue", color='black')
    plt.plot(df_ols["Year"], df_ols["PredictedRevenue"], '-o', label="OLS Prediction", color='blue')
    plt.plot(df_lasso["Year"], df_lasso["PredictedRevenue"], '-o', label="Lasso Prediction", color='red')
    plt.plot(df_ridge["Year"], df_ridge["PredictedRevenue"], '-o', label="Ridge Prediction", color='green')
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Revenue")
    plt.title("Revenue Prediction")
    plt.grid(True)
    plt.show()


def evaluate_revenue_sm_regression(df):
    return