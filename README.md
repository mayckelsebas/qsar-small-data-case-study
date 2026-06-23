# QSAR Small-Data Case Study

## Overview

This repository contains the reproducible computational workflow supporting the study:

**"Descriptor Dimensionality Versus Algorithmic Complexity in a Small-Data QSAR Case Study"**

The study evaluates whether increasing descriptor dimensionality and algorithmic complexity improve predictive performance under severe small-data conditions (n = 19). A previously published three-descriptor multiple linear regression (MLR) model is re-evaluated alongside several machine learning approaches, including Ridge Regression, Lasso Regression, Support Vector Regression (RBF kernel), Random Forest, K-Nearest Neighbours, and XGBoost.

The repository provides all files required to reproduce the reported results, figures, validation metrics, statistical tests, and Y-randomization analyses.

---

## Repository Contents

* `dataset_standard.csv` — Curated dataset containing 19 compounds and 29 molecular descriptors.
* `qsar_case_study.py` — Publication-version analysis script.
* `requirements.txt` — Python package requirements.
* `LICENSE` — Repository license.

---

## Computational Environment

The reproducibility audit was successfully validated using:

* Python 3.10
* scikit-learn 1.7.2
* XGBoost 3.2.0

Additional dependencies include NumPy, pandas, SciPy, Matplotlib, and OpenPyXL.

---

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Analysis

Execute:

```bash
python qsar_case_study.py
```

---

## Expected Outputs

The script automatically creates:

### figures/

* figure1_mlr_observed_vs_predicted.png
* figure2_mlr_residuals.png
* figure3_williams_plot.png
* figure4_xgboost_observed_vs_predicted.png
* model_comparison.png
* y_randomization.png

### results/

* model_results.csv
* svr_sensitivity.csv

---

## Reproducibility

The repository reproduces:

* LOOCV validation metrics
* Golbraikh–Tropsha external validation statistics
* Y-randomization analysis
* Statistical comparison tests
* Hyperparameter sensitivity analysis

---

## Citation

If you use this repository, please cite the associated publication.

---

## License

MIT License.
