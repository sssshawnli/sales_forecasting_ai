# Sales Revenue Forecasting Project

This project analyzes and forecasts monthly and yearly sales revenue using statistical and machine learning methods. It is designed for B2B dental supply sales data and supports model comparison, visualization, and automated evaluation.

---

## Project Structure

```
pd_project/
├── data/           # Raw input data (ORDR, RDR1, etc.)
├── notebooks/      # Jupyter notebooks for analysis and prototyping
├── src/            # Core Python modules
├── models/         # Saved models (optional)
├── output/         # Generated reports, figures, forecasts
├── main.py         # Entrypoint for batch forecasting
└── README.md       # This file
```

---

## Features

- Data loading and preprocessing from raw SAP Business One exports
- Revenue trend analysis by month, year, customer, or item
- Forecasting models: OLS, Lasso, Ridge, ARIMA, SARIMAX
- Stationarity check and differencing for time series modeling
- Model comparison based on historical fit and 2025 forecast
- Modular design using `src/` for extensibility and reusability

---

## Key Modules

| File                  | Purpose                                                  |
|-----------------------|----------------------------------------------------------|
| `src/data_load.py`    | Load and merge raw order data from ORDR and RDR1 tables  |
| `src/model.py`        | Implement forecasting models: ARIMA, Lasso, Ridge, etc.  |
| `src/evaluate.py`     | Generate visualizations and compare model predictions    |
| `notebooks/*.ipynb`   | Interactive analysis, model exploration, and documentation |

---

## Quick Start

1. Clone the repository and activate your virtual environment.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your raw CSV data (e.g., `ORDR.csv`, `RDR1.csv`) under the `data/raw/` directory.
4. Run any notebook to explore or evaluate:
   ```bash
   jupyter notebook notebooks/eva03_sales_rev_month_arima.ipynb
   ```
5. Or run the batch forecast pipeline:
   ```bash
   python main.py
   ```

---

## Output

The following outputs will be generated:

- Forecasted revenue for 2025 from each model
- Visual comparison plots: actual vs predicted revenue
- CSV files exported to the `output/` directory:
  - `ols_forecast.csv`
  - `lasso_forecast.csv`
  - `ridge_forecast.csv`

---

## Dependencies

This project requires:

- Python 3.8 or higher
- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- jupyterlab

For full details, see [requirements.txt](./requirements.txt).

---

## Notes

- Raw data is assumed to be exported from SAP Business One in CSV format.
- Models can be easily extended to include customer-level, product-level, or monthly forecasting.
- Forecasting logic is reusable via `main.py`.

---

## Author

Shawn
Developed in WSL2 + Ubuntu environment.

## Data Anonymization Notice

All customer-related information in this project has been anonymized to ensure privacy and compliance.

Identifiable fields such as CardCode, CardName, and BrandName have been replaced with non-identifying codes or pseudonyms.

No personally identifiable information (PII) is used in any part of the analysis or visualization.

This allows the dataset to be safely used for internal forecasting, modeling, and demonstration purposes without violating data privacy standards.


## License

Private internal project. Not intended for public release.
