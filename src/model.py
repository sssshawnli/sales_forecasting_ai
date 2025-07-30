# ---------------------------------------------
# Revenue Forecasting Script for Dental B2B Sales
# This script includes various forecasting models such as Linear Regression, Lasso, Ridge, ARIMA, and SARIMAX
# Each function is responsible for different forecasting scenarios (yearly, monthly, by item, by customer, etc.)
# ------

import sys
import os

# Add project root to sys.path for module resolution
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import required libraries
from sklearn.linear_model import LinearRegression
from src.data_load import load_ocrd, load_ordr, load_rdr1, load_odln, load_oitm
from src.preprocess import clean_order_date, grouped_itemcate_rev,grouped_revenue_cust,grouped_orderval_year,grouped_orderval_month,grouped_rev_year,grouped_features_month
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime 
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# Set display options for better visualization in notebooks or logs
pd.set_option("display.max_columns", None)     
pd.set_option("display.width", None)             
pd.set_option("display.float_format", '{:,.2f}'.format)  


# -------------------------------
# Predict yearly revenue using linear regression
# -------------------------------
def predict_revenue(df: pd.DataFrame, target_year: int = 2025) -> pd.DataFrame:
    """
    Predict revenue for a given target year using linear regression.
    Includes regression line, 95% confidence interval, R², and p-value.
    """
    
    # Step 1: Prepare and aggregate data by year
    df = grouped_rev_year(df)

    # Step 2: Fit OLS regression model
    x = df["Year"].values
    y = df["rev"].values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # Step 3: Predict revenue for the target year
    target_X = pd.DataFrame({"const": [1], "Year": [target_year]})
    y_pred_2025 = model.predict(target_X)[0]
    df_pred = pd.DataFrame([{"Year": target_year, "rev": y_pred_2025}])
    df_combined = pd.concat([df, df_pred], ignore_index=True)

    # Step 4: Extend data to include the prediction year for plotting
    x_all = np.append(x, target_year)
    X_all = sm.add_constant(x_all)
    y_all_pred = model.predict(X_all)
    _, ci_lower, ci_upper = wls_prediction_std(model, exog=X_all)

    # Step 5: Print regression statistics
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Intercept: {model.params[0]:.2f}")
    print(f"Slope: {model.params[1]:.2f}")
    print(f"p-value: {model.pvalues[1]:.6f}")
    
    # Step 6: Visualize results
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label="Actual Revenue")
    plt.plot(x_all, y_all_pred, "r", label="Fitted Line")
    plt.fill_between(x_all, ci_lower, ci_upper, color="gray", alpha=0.3, label="95% CI")
    plt.scatter([target_year], [y_pred_2025], color="black", marker="x", label="2025 Forecast")

    plt.title(f"Revenue Linear Fit\nR² = {model.rsquared:.3f}, p = {model.pvalues[1]:.3f}")
    plt.xlabel("Year")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    return df_combined
    
# -------------------------------
# Predict yearly revenue using OLS
# -------------------------------
def predict_revenue_sm(df: pd.DataFrame, target_year: int =2025)-> pd.DataFrame:

    """
    2025 Revenue LinearRegression
    """
    df = grouped_rev_year(df)
    x = df["Year"].values
    y = df["rev"].values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # combine 2025
    x_all = np.append(x, target_year).reshape(-1, 1)
    X_all = sm.add_constant(x_all)
    y_all_pred = model.predict(X_all)

    print(model.summary())


    # Returns a DataFrame
    df_pred = pd.DataFrame({
        "Year": x_all.flatten(),
        "PredictedRevenue": y_all_pred
    })
    df_comb = pd.merge(df, df_pred, on="Year", how="outer") 
    return df_comb
    
# -------------------------------
# Predict yearly revenue using LASSO
# -------------------------------
def predict_revenue_lasso(df: pd.DataFrame, target_year: int =2025) -> float:

    # Predict revenue using Lasso regression
    # Uses polynomial features and regularization to avoid overfitting
    # Returns a DataFrame of historical and predicted revenue
    
    df = grouped_rev_year(df)
    x = df[["Year"]].values
    y = df["rev"].values

    # Lasso regression with polynomial features (degree 2)
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LassoCV(alphas=np.logspace(-4, 4, 100), max_iter=10000, cv=3)
    )
    model.fit(x, y)

    # Generate predictions including target year
    x_all = np.append(x, target_year).reshape(-1, 1)
    y_all_pred = model.predict(x_all)

    # Returns a DataFrame of historical and predicted revenue
    df_pred = pd.DataFrame({
        "Year": x_all.flatten(),
        "PredictedRevenue": y_all_pred
    })
    df_comb = pd.merge(df, df_pred, on="Year", how="outer")  
    return df_comb

# -------------------------------
# Predict yearly revenue using Ridge
# -------------------------------
def predict_revenue_ridge(df: pd.DataFrame, target_year: int = 2025) -> float:
    

    df = grouped_rev_year(df)
    
    x = df[["Year"]].values
    y = df["rev"].values

    alphas = np.logspace(-4,4,100) # λ scope

    # Similar to Lasso but uses L2 regularization
    model = RidgeCV(alphas=alphas, cv=2)
    model.fit(x,y)

    x_all = np.append(x, target_year).reshape(-1, 1)
    y_all_pred = model.predict(x_all)

    df_pred = pd.DataFrame({
        "Year": x_all.flatten(),
        "PredictedRevenue": y_all_pred
    })
    df_comb = pd.merge(df, df_pred, on="Year", how="outer")  

# -------------------------------
# Predict revenue by item category using Linear Regression
# -------------------------------
def predict_revenue_byitem(df: pd.DataFrame, targer_year: int = 2025) -> float:

    # Fits a separate model per category

    df = grouped_itemcate_rev(df)
    category_models = {}
    category_predict = {}
    df_group = (df.groupby("Category")[["rev"]].sum())
    for cat, group in df.groupby("Category"):
        X = group["Year"].values.reshape(-1,1)
        y = group["rev"].values

        model = LinearRegression()
        model.fit(X,y)

        y_pred = model.predict(np.array([[2025]]))[0]

        category_models[cat] = model
        category_predict[cat] = y_pred

    df_pred = pd.DataFrame({
        "Category": list(category_predict.keys()),
        "Year": targer_year,
        "rev": list(category_predict.values())
    }).sort_values("rev", ascending=False)

    df_combine = pd.concat([df,df_pred],ignore_index=True)
    df_combine = df_combine.sort_values(by=(["Category","Year"]),ascending=(True,True))

    return df_combine

# -------------------------------
# Predict revenue by customer using Linear Regression
# -------------------------------
def predict_revenue_bycust(df: pd.DataFrame, target_year: int = 2025) -> float:
    """
    2025 Revenue by customer LinearRegression
    """
    df = grouped_revenue_cust(df)

    bycust_models = {}
    bycust_predict = {}

    for brandn, group in df.groupby("BrandName"):
        X = group["Year"].values.reshape(-1,1)
        y = group["rev"]

        model = LinearRegression()
        model.fit(X,y)
        
        y_pred = model.predict(np.array([[2025]]))[0]

        bycust_models[brandn] = model
        bycust_predict[brandn] = y_pred

    df_pred = pd.DataFrame({
        "BrandName": list(bycust_predict.keys()),
        "Year": target_year,
        "rev": list(bycust_predict.values())
    }).sort_values("rev",ascending=False)

    df_combine = pd.concat([df,df_pred],ignore_index=True)
    #df_combine = df_combine.sort_values(by=(["Category","Year"]),ascending=(True,True))
    

    #df_combine = df_combine[df_combine["BrandName"] =='CROSSTEX INTERNATIONAL']

    return df_pred

# -------------------------------
# Predict order volume per year using Linear Regression
# -------------------------------
def predict_order_volume(df: pd.DataFrame, target: int = 2025) -> float:

    """
    2025 order_volume LinearRegression
    """

    df = grouped_orderval_year(df)

    orderval_year_model={}
    orderval_year_predict={}

    model = LinearRegression()
    X = df["Year"].values.reshape(-1,1)
    y = df["order_valume"]

    model.fit(X,y)

    pred = model.predict(np.array([[target]]))[0]
    
    df_pred = pd.DataFrame([{
        "Year": target,
        "order_valume": pred
    }])
    df_combine = pd.concat([df,df_pred],ignore_index=True)
    return df_combine

# -------------------------------
# Predict order volume per month using Linear Regression
# -------------------------------
def predict_ordervolume_month(df: pd.DataFrame, target: int = 2025) -> float:

    """
    2025 order_volume by month LinearRegression

    """

    df = grouped_orderval_month(df)

    model = LinearRegression()
    model_ordervol = {}
    model_ordervol_pred = {}

    for month, group in df.groupby("Month"):
        X = group["Year"].values.reshape(-1,1)
        y = group["order_valume"]

        model.fit(X,y)
        y_pred = model.predict(np.array([[target]]))[0]

        model_ordervol_pred[month] = y_pred
        
    df_pred = pd.DataFrame({
        "Year" : target,
        "Month" : list(model_ordervol_pred.keys()),
        "order_valume" : list(model_ordervol_pred.values())
    })
    df_combine = pd.concat([df,df_pred],ignore_index=True)
    return df_combine

# -------------------------------
# Predict monthly revenue (more features )using Linear Regression
# -------------------------------
def predict_rev_month_mul(df: pd.DataFrame, target_date) -> float:
   
     # Step 1: Prepare and aggregate data by month
    df = grouped_features_month(df)

    # Step 2: Fit OLS regression model
    x_df = df[["Year", "Month", "order_count", "sku_count", "cust_count"]].astype(float)
    y = df["rev"].astype(float)  

    # Add intercept term
    X = sm.add_constant(x_df)
    model = sm.OLS(y, X).fit()

    # Step 3: Predict revenue for the target_date
    target_year = target_date.year
    target_month = target_date.month
    monthly_avg = df[df["Month"] == target_month][["order_count", "sku_count", "cust_count"]].mean()

    target_row = pd.DataFrame([{
        "const": 1.0,
        "Year": target_year,
        "Month": target_month,
        "order_count": monthly_avg["order_count"],
        "sku_count": monthly_avg["sku_count"],
        "cust_count": monthly_avg["cust_count"]
    }])

    target_X =sm.add_constant(target_row)

    y_pred = model.predict(target_X)[0]

    df_pred = pd.DataFrame([{
        "Year": target_year,
        "Month": target_month,
        "rev": y_pred
    }])

    df_combined = pd.concat([df, df_pred], ignore_index=True)
    print(df_combined)
     # Step 5: Print regression statistics
    print(model.summary())
    
    # Visualization
    # Step 1: build YearMonth column
    df["YearMonth"] = df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2)
    target_label = f"{target_year}-{str(target_month).zfill(2)}"

    # Step 2: append prediction
    df_pred = pd.DataFrame([{
        "Year": target_year,
        "Month": target_month,
        "rev": y_pred,
        "YearMonth": target_label
    }])

    df_combined = pd.concat([df, df_pred], ignore_index=True)

    # Step 3: 拼接历史特征 + 预测特征
    X_all = pd.concat([
        x_df,  # ✅ 真实特征数据
        target_row.drop(columns=["const"])  # ✅ 预测点（注意去掉 const）
    ], ignore_index=True)

    X_all = sm.add_constant(X_all)

    # Step 4: 预测全体拟合线
    y_fit = model.predict(X_all)

    # Step 5: 可视化
    x_axis = np.arange(len(df_combined))  # x 轴点数 = 所有 data 点数
    plt.figure(figsize=(14, 6))
    plt.plot(x_axis, df_combined["rev"], 'bo-', label='Historical Rev')
    plt.plot(x_axis, y_fit, 'g--', label='Fitted Line')
    plt.plot(x_axis[-1], y_pred, 'rx', markersize=10, label='Prediction Rev')
    plt.xticks(ticks=x_axis, labels=df_combined["YearMonth"], rotation=45)
    plt.xlabel("Year-Month")
    plt.ylabel("Revenue")
    plt.title("Monthly Revenue Forecast with OLS Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------
# Predict monthly revenue forecast (Linear OLS)"
# -------------------------------
def predict_rev_month_linear(df: pd.DataFrame, target_date) -> float:
    """
    Linear regression model for monthly revenue forecasting.
    """
    # Step 1: Feature engineering by month
    df = grouped_features_month(df)

    # Step 2: Prepare feature matrix X and target y
    features = ["Year", "Month"]
    x = df[features].astype(float)
    y = df["rev"].astype(float)
    X = sm.add_constant(x)  # Add intercept

    # Step 3: Fit OLS model
    model = sm.OLS(y, X).fit()

    # Step 4: Prepare input for prediction
    target_year = target_date.year
    target_month = target_date.month
    target_row = pd.DataFrame([{
        "const": 1.0,
        "Year": target_year,
        "Month": target_month,
    }])
    y_pred = model.predict(target_row)[0]

    # Step 5: Append predicted row and build YearMonth
    df["YearMonth"] = df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2)
    pred_label = f"{target_year}-{str(target_month).zfill(2)}"
    df_pred = pd.DataFrame([{
        "Year": target_year,
        "Month": target_month,
        "rev": y_pred,
        "YearMonth": pred_label
    }])
    df_combined = pd.concat([df, df_pred], ignore_index=True)

    # Step 6: Prediction line using same model
    X_all = sm.add_constant(df_combined[features].astype(float))
    y_fit = model.predict(X_all)

    # Step 7: Visualization
    plt.figure(figsize=(14, 6))
    x_range = np.arange(len(df_combined))
    plt.plot(x_range, df_combined["rev"], 'bo-', label='Historical Rev')
    plt.plot(x_range, y_fit, 'g--', label='Fitted Line')
    plt.plot(x_range[-1], y_pred, 'rx', markersize=10, label='Prediction Rev')
    plt.xticks(ticks=x_range, labels=df_combined["YearMonth"], rotation=45)
    plt.xlabel("Year-Month")
    plt.ylabel("Revenue")
    plt.title("Monthly Revenue Forecast (Linear OLS)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(model.summary())
    return df_combined


# -------------------------------
# Forecast monthly revenue using ARIMA and SARIMAX time series models
# -------------------------------
def predict_rev_month_arima(df: pd.DataFrame)-> float:
    
    # Step 1: Preprocess data - aggregate monthly revenue and related features
    df = grouped_features_month(df)

    # Drop non-time series columns and set DocDate as index
    df_rev = df.drop(['Year','Month'], axis=1)
    df_rev = df_rev.set_index("DocDate")

    # Step 2: Split into training data (before 2024-01) and validation data (from 2024-01)
    df_train = df_rev[df_rev.index <= '2024-1']
    df_valid = df_rev[df_rev.index >= '2024-1']
    
    # Step 3: Apply first-order differencing to revenue to achieve stationarity
    df_train["rev_diff"] = df_train["rev"].diff()

    # Drop rows with NaN values created by differencing
    df_train = df_train.dropna()
    
    # Step 4: Check for stationarity before and after differencing using Augmented Dickey-Fuller test
    df_train.plot()
    plt.title('Checking stationarity of revenue')
    plt.show()

    result_diff_before = adfuller(df_train["rev"])
    print('ADF test on revenue (before differencing):')
    print('ADF Statistic:', result_diff_before[0])
    print('p-value:', result_diff_before[1])

    result = adfuller(df_train["rev_diff"])
    print('ADF test on differenced revenue:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    # Step 5: Plot ACF and PACF to determine ARIMA parameters (manually set to (4,1,5) below)
    plot_acf(df_train["rev_diff"].dropna(), lags=18)
    plot_pacf(df_train["rev_diff"].dropna(), lags=18)
    plt.show()

    # Step 6: Fit ARIMA model (autoregressive integrated moving average)
    model = ARIMA(df_train["rev"], order=(4,1,5))
    results = model.fit()
    print('ARIMA model summary:')
    print(results.summary())

    # Forecast next 12 months with ARIMA
    forecast_result = results.forecast(steps=12)
    df_ar = forecast_result.to_frame(name = "rev")
    
    # Step 7: Prepare SARIMAX (seasonal ARIMA with exogenous variables)
    X = df_train[["order_count", "sku_count", "cust_count"]]
    y = df_train["rev"]

    # With exogenous features
    seasonal_model_exog = SARIMAX(
        endog=y,
        exog=X,
        order=(4,1,5), seasonal_order=(1,1,1,12))

    # Without exogenous features
    seasonal_model_noexog = SARIMAX(df_train["rev"], order=(4,1,5), seasonal_order=(1,1,1,12))
    
    # Fit both SARIMA models
    results_seasonal_model_exog = seasonal_model_exog.fit()
    print('SARIMAX with exogenous variables summary:')
    print(results_seasonal_model_exog.summary())

    results_seasonal_model_noexog = seasonal_model_noexog.fit()
    print('SARIMA without exogenous variables summary:')
    print(results_seasonal_model_noexog.summary())

    # Step 8: Forecast future exogenous features using separate ARIMA models
    X_futrue = pd.DataFrame(index=pd.date_range("2024-02-01", periods=12, freq="MS"))
    for col in ["order_count", "sku_count", "cust_count"]:
        model = ARIMA(df_train[col], order=(4,1,5))
        result = model.fit()
        forecast = result.forecast(steps=12)
        X_futrue[col] = forecast.values

    # Step 9: Use fitted SARIMA models to forecast next 12 months
    forecast_seasonal_model_exog = results_seasonal_model_exog.get_forecast(steps=12, exog=X_futrue)
    fs_exog = forecast_seasonal_model_exog.summary_frame()

    forecast_seasonal_model_noexog = results_seasonal_model_noexog.get_forecast(steps=12)
    fs_noexog = forecast_seasonal_model_noexog.summary_frame()

    # Step 10: Visualize SARIMA (with exogenous) vs ARIMA forecast
    plt.figure(figsize=(14,6))
    plt.plot(df_train.index.to_timestamp(), df_train["rev"], marker="o", label='Training Data')
    plt.plot(df_valid.index.to_timestamp(), df_valid["rev"], marker="o", label='Validation Data')
    plt.plot(fs_exog.index.to_timestamp(), fs_exog["mean"], marker="o", label='SARIMA Forecast')
    plt.plot(df_ar.index.to_timestamp(), df_ar["rev"], marker="o", label='ARIMA Forecast')
    plt.fill_between(fs_exog.index.to_timestamp(), fs_exog["mean_ci_lower"], fs_exog["mean_ci_upper"], color="gray", alpha=0.3, label="95% CI")
    plt.title("SARIMAX Time Series Monthly Revenue Forecasting (with exogenous variables)")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xlabel("Date")
    plt.ylabel("Monthly Revenue")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Step 11: Visualize SARIMA (without exogenous) vs ARIMA forecast
    plt.figure(figsize=(14,6))
    plt.plot(df_train.index.to_timestamp(), df_train["rev"], marker="o", label='Training Data')
    plt.plot(df_valid.index.to_timestamp(), df_valid["rev"], marker="o", label='Validation Data')
    plt.plot(fs_noexog.index.to_timestamp(), fs_noexog["mean"], marker="o", label='SARIMA Forecast')
    plt.plot(df_ar.index.to_timestamp(), df_ar["rev"], marker="o", label='ARIMA Forecast')
    plt.fill_between(fs_noexog.index.to_timestamp(), fs_noexog["mean_ci_lower"], fs_noexog["mean_ci_upper"], color="gray", alpha=0.3, label="95% CI")
    plt.title("SARIMA Time Series Monthly Revenue Forecasting (no exogenous variables)")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xlabel("Date")
    plt.ylabel("Monthly Revenue")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    return  