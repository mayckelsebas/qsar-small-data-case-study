# =============================================================
# qsar_case_study.py
# Zenodo Reproducibility Release
# =============================================================
# Input:
#   dataset_standard.csv
#
# Outputs:
#   results/
#   figures/
#
# Compatible with:
#   Python 3.10
#   scikit-learn 1.7.2
#   xgboost 3.2.0
# =============================================================

import os
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ==============================================================
# QSAR vs ML Small-Data Benchmark
# UPDATED REVIEWER-ALIGNED VERSION (Google Colab)
# ==============================================================
#
# PURPOSE OF THIS VERSION
# -----------------------
# This script implements minimal methodological corrections
# requested during peer review while preserving the original
# scope and framing of the study as a constrained
# methodological case study.
#
# MAIN IMPROVEMENTS
# -----------------
# ✔ Proper preprocessing inside LOOCV using pipelines
# ✔ Explicit reproducibility controls
# ✔ Removal of redundant Q²_F1 metric
# ✔ Y-randomization for BOTH:
#       - Classical MLR
#       - Best ML model
# ✔ Minimal hyperparameter sensitivity analysis
# ✔ Publication-ready figures
#
# COMPATIBLE WITH:
# ✔ Google Colab
# ✔ Python >= 3.10
#
# ==============================================================

# =============================
# 1. INSTALL DEPENDENCIES
# =============================

# Uncomment in Google Colab if needed
# !pip install xgboost

# =============================
# 2. IMPORT LIBRARIES
# =============================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from scipy import stats

import xgboost as xgb

# =============================
# 3. REPRODUCIBILITY
# =============================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# =============================
# 4. LOAD DATASET
# =============================

# Upload your dataset to Colab first


df = pd.read_csv("dataset_standard.csv")

print("="*60)
print("DATASET INFORMATION")
print("="*60)
print("Dataset shape:", df.shape)

if "Log Ki" not in df.columns:
    raise ValueError("Column 'Log Ki' not found.")

# =============================
# 5. DEFINE FEATURES
# =============================

y = df["Log Ki"]

X_full = df.drop(columns=["Log Ki"])

# Classical descriptor subset
selected_descriptors = ["Mi", "nCs", "nHDon"]

X_selected = X_full[selected_descriptors]

print("\nSelected descriptors (MLR):")
print(selected_descriptors)

# =============================
# 6. CROSS-VALIDATION
# =============================

cv = LeaveOneOut()

# =============================
# 7. DEFINE MODELS
# =============================

models = {

    "Ridge": Ridge(alpha=1.0),

    "Lasso": Lasso(
        alpha=0.01,
        max_iter=10000
    ),

    "SVR_RBF": SVR(
        kernel="rbf",
        C=1.0,
        epsilon=0.1
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=RANDOM_STATE
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=3
    ),

    "XGBoost": xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=1
    )
}

# =============================
# 8. METRIC FUNCTION
# =============================

def qsar_metrics(y_true, y_pred):

    residuals = y_true - y_pred

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    # PRESS
    press = np.sum(residuals**2)

    # Mean of observed values
    y_mean = np.mean(y_true)

    # Q²_F2
    q2_f2 = 1 - (
        press /
        np.sum((y_pred - y_mean)**2)
    )

    return {
        "R2_LOOCV": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Q2_F2": q2_f2
    }

# =============================
# GT-TYPE VALIDATION METRICS
# =============================

#Bloque insertado después de cuestionar el valor de GT, no estaba en el código inicial

def golbraikh_tropsha_metrics(y_true, y_pred):

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Standard R²
    r2 = r2_score(y_true, y_pred)

    # Regression through origin
    k = np.sum(y_true * y_pred) / np.sum(y_pred**2)

    # Predicted values adjusted through origin
    y_pred_origin = k * y_pred

    # R²₀ calculation
    ss_res_0 = np.sum((y_true - y_pred_origin)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)

    r2_0 = 1 - (ss_res_0 / ss_tot)

    # GT deviation ratio
    gt_ratio = np.abs(r2 - r2_0) / r2

    return {
        "k_GT": k,
        "R2_0_GT": r2_0,
        "GT_ratio": gt_ratio
    }


# =============================
# 9. CLASSICAL MLR MODEL
# =============================

print("\n" + "="*60)
print("CLASSICAL MLR MODEL")
print("="*60)

pipeline_classical = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

y_pred_classical = cross_val_predict(
    pipeline_classical,
    X_selected,
    y,
    cv=cv
)

metrics_classical = qsar_metrics(
    y,
    y_pred_classical
)

# GT metrics for MLR
gt_classical = golbraikh_tropsha_metrics(
    y,
    y_pred_classical
)

print("\nGT Metrics (MLR)")
for k, v in gt_classical.items():
    print(f"{k}: {v:.4f}")

for k, v in metrics_classical.items():
    print(f"{k}: {v:.4f}")

# =============================
# FIGURE 1
# OBSERVED VS PREDICTED (MLR)
# =============================

plt.figure(figsize=(6,6))

plt.scatter(
    y,
    y_pred_classical,
    s=80
)

min_val = min(y.min(), y_pred_classical.min())
max_val = max(y.max(), y_pred_classical.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Observed Log Ki")
plt.ylabel("Predicted Log Ki")

plt.title(
    "Observed vs Predicted (Classical MLR)"
)

plt.tight_layout()

plt.savefig(
    "figures/figure1_mlr_observed_vs_predicted.png",
    dpi=300
)


# =============================
# FIGURE 2
# RESIDUALS VS PREDICTED (MLR)
# =============================

residuals_mlr = y - y_pred_classical

plt.figure(figsize=(6,6))

plt.scatter(
    y_pred_classical,
    residuals_mlr,
    s=80
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Log Ki")
plt.ylabel("Residuals")

plt.title(
    "Residuals vs Predicted (Classical MLR)"
)

plt.tight_layout()

plt.savefig(
    "figures/figure2_mlr_residuals.png",
    dpi=300
)


# =============================
# FIGURE 3
# WILLIAMS PLOT (MLR)
# =============================

# Fit classical MLR on full reduced descriptor set
pipeline_classical.fit(X_selected, y)

# Extract scaled descriptors
X_scaled = pipeline_classical.named_steps["scaler"].transform(X_selected)

# Add intercept term
X_matrix = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))

# Hat matrix
H = X_matrix @ np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T

# Leverage values
leverage = np.diag(H)

# Standardized residuals
std_residuals = residuals_mlr / np.std(residuals_mlr)

# Critical leverage threshold
p = X_selected.shape[1]
n = X_selected.shape[0]

h_star = 3 * (p + 1) / n

# Plot
plt.figure(figsize=(7,6))

plt.scatter(
    leverage,
    std_residuals,
    s=80
)

plt.axhline(
    y=3,
    linestyle="--"
)

plt.axhline(
    y=-3,
    linestyle="--"
)

plt.axvline(
    x=h_star,
    linestyle="--"
)

plt.xlabel("Leverage (hᵢ)")
plt.ylabel("Standardized Residuals")

plt.title("Williams Plot (Classical MLR)")

plt.tight_layout()

plt.savefig(
    "figures/figure3_williams_plot.png",
    dpi=300
)


print(f"Critical leverage (h*): {h_star:.4f}")

# =============================
# 10. FULL DESCRIPTOR MODELS
# =============================

print("\n" + "="*60)
print("MACHINE LEARNING MODELS")
print("="*60)

results = {}
predictions = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    y_pred = cross_val_predict(
        pipeline,
        X_full,
        y,
        cv=cv
    )

    metrics = qsar_metrics(y, y_pred)

    results[name] = metrics
    predictions[name] = y_pred

    print(f"\n{name}")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

# =============================
# 11. RESULTS TABLE
# =============================

# Add GT metrics to results

for model_name in results.keys():

    y_pred_model = predictions[model_name]

    gt_metrics = golbraikh_tropsha_metrics(
        y,
        y_pred_model
    )

    results[model_name].update(gt_metrics)

results_df = pd.DataFrame(results).T

print("\n" + "="*60)
print("SUMMARY RESULTS")
print("="*60)

# Add classical MLR results to summary table

results["Classical_MLR"] = {
    **metrics_classical,
    **gt_classical
}

results_df = pd.DataFrame(results).T

print(results_df)

# Save table
results_df.to_csv("results/model_results.csv")


# =============================
# 12. BEST MODEL SELECTION
# =============================

# Select best ML model only
ml_results_df = results_df.drop(index="Classical_MLR")

best_model_name = ml_results_df["R2_LOOCV"].idxmax()

best_model = models[best_model_name]

best_metrics = results[best_model_name]

y_pred_best = predictions[best_model_name]

# GT metrics for best ML model
gt_best = golbraikh_tropsha_metrics(
    y,
    y_pred_best
)

print("\nGT Metrics (Best ML)")
for k, v in gt_best.items():
    print(f"{k}: {v:.4f}")

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)

print("Best model:", best_model_name)

for k, v in best_metrics.items():
    print(f"{k}: {v:.4f}")

# =============================
# 13. PREDICTED VS OBSERVED PLOT
# =============================

plt.figure(figsize=(6,6))

plt.scatter(
    y,
    y_pred_best,
    s=80
)

min_val = min(y.min(), y_pred_best.min())
max_val = max(y.max(), y_pred_best.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Observed Log Ki")
plt.ylabel("Predicted Log Ki")

plt.title(
    f"Observed vs Predicted ({best_model_name})"
)

plt.tight_layout()

plt.savefig(
    "figures/figure4_xgboost_observed_vs_predicted.png",
    dpi=300
)


# =============================
# 14. MODEL COMPARISON PLOT
# =============================

plt.figure(figsize=(10,5))

results_df["R2_LOOCV"].sort_values().plot(
    kind="barh"
)

plt.xlabel("R² LOOCV")
plt.title("Model Performance Comparison")

plt.tight_layout()

plt.savefig(
    "figures/model_comparison.png",
    dpi=300
)


# =============================
# 15. Y-RANDOMIZATION (MLR)
# =============================

print("\n" + "="*60)
print("Y-RANDOMIZATION - MLR")
print("="*60)

n_permutations = 1000

random_r2_mlr = []

pipeline_mlr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

real_r2_mlr = metrics_classical["R2_LOOCV"]

for i in range(n_permutations):

    y_random = np.random.permutation(y)

    y_pred_random = cross_val_predict(
        pipeline_mlr,
        X_selected,
        y_random,
        cv=cv
    )

    r2_random = r2_score(
        y_random,
        y_pred_random
    )

    random_r2_mlr.append(r2_random)

random_r2_mlr = np.array(random_r2_mlr)

z_score_mlr = (
    real_r2_mlr - np.mean(random_r2_mlr)
) / np.std(random_r2_mlr)

print("Real R²:", round(real_r2_mlr,4))
print("Random mean R²:", round(np.mean(random_r2_mlr),4))
print("Z-score:", round(z_score_mlr,4))

# =============================
# 16. Y-RANDOMIZATION (BEST ML)
# =============================

print("\n" + "="*60)
print("Y-RANDOMIZATION - BEST ML MODEL")
print("="*60)

random_r2_ml = []

pipeline_best = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_model)
])

real_r2_ml = best_metrics["R2_LOOCV"]

for i in range(n_permutations):

    y_random = np.random.permutation(y)

    y_pred_random = cross_val_predict(
        pipeline_best,
        X_full,
        y_random,
        cv=cv
    )

    r2_random = r2_score(
        y_random,
        y_pred_random
    )

    random_r2_ml.append(r2_random)

random_r2_ml = np.array(random_r2_ml)

z_score_ml = (
    real_r2_ml - np.mean(random_r2_ml)
) / np.std(random_r2_ml)

print("Real R²:", round(real_r2_ml,4))
print("Random mean R²:", round(np.mean(random_r2_ml),4))
print("Z-score:", round(z_score_ml,4))

# =============================
# 17. Y-RANDOMIZATION PLOTS
# =============================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12,5)
)

# --- MLR

axes[0].hist(
    random_r2_mlr,
    bins=30
)

axes[0].axvline(
    real_r2_mlr,
    linestyle="--"
)

axes[0].set_title("Y-Randomization (MLR)")
axes[0].set_xlabel("Randomized R²")

# --- ML

axes[1].hist(
    random_r2_ml,
    bins=30
)

axes[1].axvline(
    real_r2_ml,
    linestyle="--"
)

axes[1].set_title(
    f"Y-Randomization ({best_model_name})"
)

axes[1].set_xlabel("Randomized R²")

plt.tight_layout()

plt.savefig(
    "figures/y_randomization.png",
    dpi=300
)


# =============================
# 18. STATISTICAL COMPARISON
# =============================

print("\n" + "="*60)
print("STATISTICAL COMPARISON")
print("="*60)

errors_classical = np.abs(y - y_pred_classical)

errors_best = np.abs(y - y_pred_best)

# Paired t-test
t_stat, p_value_t = stats.ttest_rel(
    errors_classical,
    errors_best
)

# Wilcoxon
w_stat, p_value_w = stats.wilcoxon(
    errors_classical,
    errors_best
)

print("Paired t-test p-value:", round(p_value_t,4))
print("Wilcoxon p-value:", round(p_value_w,4))

# =============================
# 19. MINIMAL HYPERPARAMETER
#    SENSITIVITY ANALYSIS
# =============================

print("\n" + "="*60)
print("MINIMAL HYPERPARAMETER SENSITIVITY")
print("="*60)

svr_results = []

for C in [0.1, 1, 10]:

    for epsilon in [0.01, 0.1]:

        model = SVR(
            kernel="rbf",
            C=C,
            epsilon=epsilon
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        y_pred = cross_val_predict(
            pipeline,
            X_full,
            y,
            cv=cv
        )

        r2 = r2_score(y, y_pred)

        svr_results.append({
            "C": C,
            "epsilon": epsilon,
            "R2_LOOCV": r2
        })

svr_df = pd.DataFrame(svr_results)

print("\nSVR Sensitivity Results")
print(svr_df)

svr_df.to_csv(
    "results/svr_sensitivity.csv",
    index=False
)

# =============================
# 20. FINAL SUMMARY
# =============================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print("\nClassical MLR:")
for k, v in metrics_classical.items():
    print(f"{k}: {v:.4f}")

print("\nBest ML Model:", best_model_name)
for k, v in best_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nY-Randomization:")
print(f"MLR Z-score: {z_score_mlr:.4f}")
print(f"{best_model_name} Z-score: {z_score_ml:.4f}")

print("\nFiles generated:")
print("- model_results.csv")
print("- svr_sensitivity.csv")
print("- observed_vs_predicted.png")
print("- model_comparison.png")
print("- y_randomization.png")

print("\nAnalysis completed successfully.")

