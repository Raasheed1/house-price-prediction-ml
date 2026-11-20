# House Price Prediction

An end-to-end regression project that predicts house prices from property features using Python and scikit-learn. The focus is on doing it **correctly**: leakage-safe preprocessing, honest cross-validated evaluation, and documenting every experiment, including the ones that did not help.

> **Final model:** Linear Regression on log(price) with log(area), 12 features.
> **Test R² = 0.663**, MAE ≈ 785K (about 16% of the average price), on 164 unseen houses.

## Highlights

- Rebuilt the preprocessing of an earlier attempt to remove data leakage (the scaler had been fitted before the train-test split, and the target was standardized unnecessarily).
- Ran 25+ experiments covering encoding, outlier handling, scaling, transformations, feature engineering, feature selection, regularization, tree ensembles, hyperparameter tuning and a blended model.
- Evaluated everything with Repeated 5-fold cross-validation on the training set, and used **nested cross-validation** for tuned models so the scores are not optimistic.
- Kept the 164-house test set for the final evaluation only.
- Wrote a detailed report explaining every method, why it was kept or rejected, the maths, and the results (see `reports/`).

## Dataset

`Housing.csv`: 545 houses, 12 features plus the target `price`. No missing values and no duplicate rows.

| Column | Description |
|---|---|
| `price` | House price (target) |
| `area` | Property area |
| `bedrooms`, `bathrooms`, `stories`, `parking` | Counts |
| `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` | Yes/no flags |
| `furnishingstatus` | Furnished, semi-furnished or unfurnished |

Source: **TODO: add the link to where you downloaded the dataset, and check its license before committing the CSV.**

## Approach

| Stage | What was done |
|---|---|
| Data inspection | Shape, types, missing values, duplicates, category spelling |
| Encoding | Yes/no as 0/1; furnishing as ordinal (compared with one-hot) |
| Outliers | IQR and Z-score detection; removal and capping tested inside each fold |
| Scaling | Four scalers tested inside Pipelines (no effect on plain Linear Regression, as expected) |
| Transformations | Log, Box-Cox and Yeo-Johnson on the target; log on `area` |
| EDA | Correlations, multicollinearity, scatter and box plots (training data only) |
| Feature engineering | Ratios, interactions and categories; none helped |
| Feature selection | Drop-one ablation and SelectKBest; all 12 features kept |
| Models | Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, blend |
| Tuning | RandomizedSearchCV inside nested cross-validation |
| Final evaluation | One-time test-set evaluation with residual analysis |

## Results

Cross-validated results (Repeated 5-fold x 3, training set only):

| Model | CV R² (mean ± std) | MAE |
|---|---|---|
| **Linear (log price, log area)** | **0.6616 ± 0.0599** | 773,038 |
| Lasso | 0.6612 ± 0.0593 | 773,026 |
| Ridge | 0.6608 ± 0.0576 | 769,480 |
| ElasticNet | 0.6601 ± 0.0575 | 771,656 |
| Blend: Linear + tuned Gradient Boosting | 0.6578 ± 0.0504 | 758,497 |
| Plain Linear (raw price, raw area) | 0.6394 ± 0.0571 | 811,162 |
| Gradient Boosting (tuned, nested) | 0.6139 ± 0.0461 | 789,231 |
| Random Forest (tuned, nested) | 0.5869 ± 0.0581 | 808,872 |

Held-out test set (164 houses):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| First model (raw price, plain linear) | 0.6704 | 780,014 | 1,068,603 |
| **Final model (log price + log area)** | **0.6630** | 784,966 | 1,080,633 |

## Key findings

- **The log transform of the target was the only change that clearly helped** (CV R² 0.639 to 0.662).
- Scaling, outlier removal, one-hot encoding, engineered features, feature selection and regularization gave no reliable improvement.
- Tree ensembles scored below the linear model, even after tuning, probably because the relationships are close to additive on the log scale and the dataset is small.
- The test R² (0.663) matches the cross-validated estimate (0.662). The first model scored slightly higher on this one test split (0.670), but the difference is well within the noise of a 164-house sample.

## Limitations

- R² of about 0.66 means roughly a third of price variation is unexplained. The file has no location detail, house age or condition, which are likely important.
- Only 545 rows, so scores are noisy; the test results come from a single split.
- A typical error of about 16% of the average price is fine for a rough estimate, not for pricing a sale.
- Predictions from the log model are closer to median prices, which can under-predict the most expensive houses.

## Repository structure

```
house-price-prediction-ml/
├── data/
│   └── Housing.csv
├── notebooks/
│   └── house_price_prediction.ipynb     # TODO: use your notebook's real file name
├── reports/
│   └── House_Price_Prediction_Report.pdf
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Getting started

```bash
git clone https://github.com/TODO-your-username/house-price-prediction-ml.git
cd house-price-prediction-ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/house_price_prediction.ipynb
```

## Report

The full write-up, with every stage, method, formula and experiment, is in [`reports/House_Price_Prediction_Report.pdf`](reports/House_Price_Prediction_Report.pdf).

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter Notebook

## Author

**Raasheed Muhammad**
LinkedIn: TODO-add-your-LinkedIn-URL | GitHub: TODO-add-your-GitHub-URL
