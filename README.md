# Sales Revenue Forecasting Project

This project analyzes and forecasts monthly and yearly sales revenue using statistical and machine learning models.
It is designed for B2B dental supply sales data and supports model comparison, visualization, and automated evaluation.

---

## Project Structure

```
pd_project/
├── data/           # Raw input data (ORDR, RDR1, etc.)
├── docs/           # Project report
├── notebooks/      # Jupyter notebooks for EDA and modeling
├── src/            # Core Python modules
├── models/         # Saved models (optional)
├── output/         # Generated plots, forecasts, reports
├── main.py         # Entrypoint for batch forecasting
└── README.md       # Project documentation
```

---

## Data Source

The raw datasets used for model training and forecasting are stored in Google Drive:
[Access the data here](https://drive.google.com/drive/folders/1VLoFvVjqo7XCLlRdd4cuSTxUGT18Wi_Y?usp=drive_link)

Place the downloaded `.csv` files (e.g., `ORDR.csv`, `RDR1.csv`) into:

```
data/raw/
```

---

## Features

- Data loading and preprocessing from SAP Business One exports
- Revenue trend analysis by month, year, customer, and item
- Forecasting models: OLS, Lasso, Ridge, ARIMA, SARIMAX
- Stationarity checks and differencing for time series modeling
- Model comparison with 2025 forecast results
- Modular pipeline under `src/` for extensibility

---

## Key Modules

| File                   | Purpose                                                  |
|------------------------|----------------------------------------------------------|
| `src/data_load.py`     | Load and merge raw ORDR and RDR1 CSV files               |
| `src/model.py`         | Define OLS, Ridge, Lasso, ARIMA, and SARIMAX models      |
| `src/evaluate.py`      | Evaluate models and create visualizations                |
| `notebooks/*.ipynb`    | Interactive exploration and documentation                |

---

## Quick Start

1. Clone the repository and activate your virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place raw data (e.g., `ORDR.csv`, `RDR1.csv`) into:
   ```
   data/raw/
   ```

4. Run analysis in Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/eva03_sales_rev_month_arima.ipynb
   ```

5. Or run batch forecast pipeline:
   ```bash
   python main.py
   ```

---

## Output

The following outputs will be generated in the `output/` directory:

- Forecasted revenue for 2025 from each model
- Visual plots comparing actual vs predicted revenue

---

## Dependencies

This project requires:

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- jupyterlab

Full list in: [requirements.txt](./requirements.txt)

---

## Notes

- Raw data is assumed to be exported from SAP Business One as `.csv`.
- Forecasting logic can be extended to:
  - Customer-level revenue
  - Product-level trends
  - Quarterly or seasonal decomposition
- Pipeline logic is reusable via `main.py`

---

## Author

Shawn  
Developed using WSL2 + Ubuntu environment.

---

## Data Anonymization Notice

All customer-related fields in this project have been anonymized.
Fields like `CardCode`, `CardName`, and `BrandName` have been pseudonymized to ensure:

- Privacy compliance
- Safe internal sharing
- No PII (personally identifiable information) is exposed

---

## License

Private Internal Project  
Not intended for public distribution or reuse.
