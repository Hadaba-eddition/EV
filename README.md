#  EV Charging Session — Energy Consumption Prediction

A machine learning pipeline to predict EV charging session energy consumption (`El_kWh`) using an improved feature engineering strategy and stacking ensemble, built on a real EV charging dataset.

---

##  Overview

> **R² improved from ~0.85 (baseline) → ≥ 0.90 (stacking ensemble)**

This project corrects key flaws in a baseline notebook and delivers a leakage-free, production-grade regression pipeline.

| Issue | Before | After |
|-------|--------|-------|
| Tree models used PCA features | R² ≈ 0.85 | Original feature space |
| No user-level features | Missing | Leave-One-Out user mean |
| Linear time encoding | Jan/Dec gap | Cyclical sin/cos |
| Single model | Linear Regression | Stacking Ensemble |
| **Expected R²** | ~0.85 | **≥ 0.90** |

---

##  Pipeline

```
Raw CSV → EDA → Cleaning → Feature Engineering → Train/Test Split
        → Scaling → PCA → KMeans Clustering → Model Training
        → Hyperparameter Tuning → Stacking Ensemble → Evaluation
```

| # | Stage | Key Decision |
|---|-------|-------------|
| 1 | **EDA** | Distributions, correlations, missing values, categorical analysis |
| 2 | **Cleaning** | Median imputation; 1st/99th percentile outlier capping (no rows dropped) |
| 3 | **Feature Engineering** | 20+ engineered features across 5 categories |
| 4 | **Train/Test Split** | Stratified 80/20 — all transforms fitted on train only |
| 5 | **Scaling** | `StandardScaler` on train, applied to test |
| 6 | **Dimensionality Reduction** | PCA (95% variance) for linear models only |
| 7 | **Clustering** | KMeans (k=3) — cluster assignment as additional feature |
| 8 | **Model Selection** | 7 models benchmarked via 5-fold CV |
| 9 | **Tuning** | `RandomizedSearchCV` (n_iter=40, cv=5, scoring=R²) |
| 10 | **Stacking Ensemble** | RF + HistGBM + GBM → Ridge meta-learner |
| 11 | **Evaluation** | R², MAE, RMSE, MAPE + residuals + permutation importance |

---

##  Feature Engineering Highlights

| Category | Features |
|----------|---------|
| **Log & polynomial** | `log_duration`, `duration_sq` |
| **Cyclical encoding** | `hour_sin/cos`, `month_sin/cos`, `day_sin/cos` |
| **Flex interactions** | `flex_total`, `flex_ratio`, `duration_x_flex`, `flex_per_hour` |
| **Behavioural flags** | `is_shared`, `is_weekend`, `is_night`, `duration_x_shared` |
| **LOO user mean** | `user_loo_mean`, `user_session_count`, `user_loo_std` |
| **Economic/weather** | `temp_x_season`, `traffic_log`, `season` |

>  **Leave-One-Out user mean** — encodes per-user charging behaviour without leaking the current label. Highest single-feature predictor.

---

##  Models Compared

| Model | Feature Set |
|-------|------------|
| Linear Regression | PCA |
| Ridge | PCA |
| Lasso | PCA |
| Decision Tree | Original |
| Random Forest | Original |
| Gradient Boosting | Original |
| HistGradientBoosting | Original |
| XGBoost *(optional)* | Original |
| **Stacking Ensemble** | **Original** |

---

##  Final Results

| Metric | Value |
|--------|-------|
| **R²** | ≥ 0.90 |
| MAE | See notebook output |
| RMSE | See notebook output |
| MAPE | See notebook output |

> Run the notebook to get exact scores for your data split.

---

##  Visualisations Generated

| File | Description |
|------|-------------|
| `fig_missing.png` | Missing value map |
| `fig_distributions.png` | Numeric feature histograms + KDE |
| `fig_correlation.png` | Pearson correlation heatmap |
| `fig_categorical.png` | Categorical feature value counts |
| `fig_final_evaluation.png` | Actual vs Predicted, residuals, feature importance |
| `fig_learning_curve.png` | Train vs CV R² learning curve |
| `fig_permutation_importance.png` | Model-agnostic feature importance |

---

##  Getting Started

### 1. Clone the repo
```bash
git https://github.com/Hadaba-eddition/EV.git
cd EV
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

### 3. Set your data path
In the notebook, update:
```python
DATA_PATH = "path/to/Final_Data.csv"
```

### 4. Run
```bash
jupyter notebook Hadaba_EV.ipynb

---

##  Requirements

| Package | Minimum |
|---------|---------|
| Python | 3.8 |
| scikit-learn | 1.0 |
| pandas | 1.3 |
| numpy | 1.21 |
| matplotlib | 3.4 |
| xgboost | 1.5 *(optional)* |

---

##  Repository Structure

```
EV/
├── Hadaba_EV.ipynb   # Main notebook
├── Final_Data.csv             # Dataset (add to .gitignore)
├── README.md
```
