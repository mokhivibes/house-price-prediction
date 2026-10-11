# House Price Prediction

Predicting house sale prices in Ames, Iowa with scikit-learn, XGBoost, LightGBM and CatBoost: a full pipeline from data exploration to a live demo.

## Problem

One row represents one house that was sold in Ames, Iowa. The columns are the features that describe it, such as `YearBuilt` and `GrLivArea`. The target is `SalePrice`, which is a continuous number, so this is a regression problem.

## Data

- Source: Ames Housing, downloaded with `fetch_openml(name="house_prices")`
- Size: 1,460 houses and 81 columns
- Saved locally as `data/ames_housing.csv`

## Metric

The main metric is **MAE** (mean absolute error): the average size of the model's mistakes in dollars. It is easy to explain to anyone, for example "the model is off by about $18,000 on average" (an illustrative figure, not a result). I also report **RMSE**, which punishes large mistakes more heavily, and **R²**, which shows how much of the price variation the model explains.

## Plan

1. Explore and split the data
2. Build a cleaning pipeline (imputer, scaler, encoder)
3. Baseline model, then compare several models with cross-validation
4. Tune, try boosting libraries and stacking
5. Evaluate once on the test set, explain the results, build a demo

## Results

All scores are 5-fold cross-validation on the training set. The test set has not been used yet.

| Model | Training data | MAE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression (first baseline) | all 1,168 houses, all columns | $20,457 | $37,871 | 0.756 |
| Linear Regression (cleaned baseline) | 1,166 houses, `PoolArea` and `GarageYrBlt` dropped | $17,207 | $25,396 | 0.890 |

The cleaning decisions behind the second row, and the experiments that justify them, are in `03_eda.ipynb` and `05_cleaning_evidence.ipynb`.
## Status

Work in progress.

