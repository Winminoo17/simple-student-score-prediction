# Student Score Prediction with Machine Learning

A hands-on machine learning notebook that uses a small dataset of student scores to practice **regression**, **binary classification**, and **multi-class classification** with scikit-learn.

## Overview

The dataset contains 25 students with four scores each: **Quiz**, **Lab**, **Midterm**, and **Final**. The notebook builds three kinds of models on this data:

| Task | Goal | Model |
|------|------|-------|
| Regression | Predict the **Final exam score** from Quiz, Lab, and Midterm | Linear Regression |
| Binary classification | Predict **Pass / Fail** (total score ≥ 50) | Logistic Regression |
| Multi-class classification | Predict **Outstanding / Satisfactory / Unsatisfactory** | Logistic Regression, Decision Tree, Random Forest, XGBoost |

## What the notebook covers

1. **Data preparation**: building a pandas DataFrame and creating `Total`, `Pass`, and `Level` columns.
2. **Linear Regression**
   - Model 1 uses all three features (Quiz, Lab, Midterm).
   - Model 2 uses only the most important feature.
   - The two models are compared with MSE, RMSE, MAE, and R².
3. **Logistic Regression** for pass/fail prediction, evaluated with accuracy, precision, recall, F1, and a classification report.
4. **Tree-based learning** for three performance levels, with macro-averaged metrics.
5. **Visualization** of predicted vs. actual values for the most important feature.

## Tech stack

- Python 3
- NumPy, pandas
- Matplotlib, seaborn
- scikit-learn
- XGBoost

## Getting started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter

# 3. Launch the notebook
jupyter notebook <notebook-name>.ipynb
```

You can also upload the notebook to [Google Colab](https://colab.research.google.com) and run it there.

## Key findings

- In the regression task, the model using all three features performed better than the single-feature model, with lower error (MSE, RMSE, MAE) and a higher R².
- Quiz score had the largest coefficient, so it was treated as the most influential feature.
- All models are evaluated on the same data they were trained on, so the scores show how well the model fits, not how well it generalizes.

## Limitations and next steps

- The dataset is very small (25 rows), so results are illustrative only.
- Add a **train/test split** or **cross-validation** to measure real generalization.
- Standardize features before logistic regression.
- Tune hyperparameters and compare models fairly on held-out data.


