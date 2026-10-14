"""
STEP 2 - Task 2.1: predict the GOAL DIFFERENCE of a match (Team A goals - Team B goals).

104 rows (one per match), 8 explanatory variables known before kick-off.
  Part A: the required 8-variable linear regression (coefficients, R2, checks)
  Part B: compare it with other models using cross-validation (honest out-of-sample test)
Results are saved to outputs/results/.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model_helpers import compare_models, fit_ols

data = pd.read_csv("../data/matches_104.csv")
# 4 variables shared with Task 2.2 (the brief allows at most 4) + 4 used only here
features = ["rank_diff", "host_advantage", "same_confederation", "stage_num",
            "titles_diff", "rest_diff", "points_diff", "conf_strength_diff"]
target = "goal_diff"

# ---- Part A: the 8-variable linear regression ----
table, checks = fit_ols(data, features, target)
table.to_csv("../outputs/results/task21_ols_coefficients.csv")
print(table.round(3))
print({k: round(float(v), 3) for k, v in checks.items()})

# Extra checks used in the report: are predictors too similar to each other? do outliers matter?
correlations = data[features].corr().abs().values - np.eye(len(features))
checks["max_correlation_between_predictors"] = correlations.max()
lr = LinearRegression().fit(data[features], data[target])
residuals = data[target] - lr.predict(data[features])
keep = residuals.abs() < 2.5 * residuals.std()          # drop matches with unusually large errors
lr_no_outliers = LinearRegression().fit(data.loc[keep, features], data.loc[keep, target])
checks["outliers_removed"] = int((~keep).sum())
checks["rank_coef_all_rows"] = lr.coef_[0]
checks["rank_coef_without_outliers"] = lr_no_outliers.coef_[0]

# ---- Part B: compare models with repeated 5-fold cross-validation ----
models = {
    "Baseline (mean)":   (DummyRegressor(), features),
    "OLS (rank only)":   (LinearRegression(), ["rank_diff"]),
    "OLS (8 variables)": (LinearRegression(), features),
    "Ridge":             (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), features),
    "Lasso":             (make_pipeline(StandardScaler(), LassoCV(cv=5)), features),
    "Random forest":     (RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=0), features),
    "Gradient boosting": (GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0), features),
}
comparison, predictions = compare_models(data, features, target, models)
comparison.to_csv("../outputs/results/task21_model_comparison.csv", index=False)
print(comparison.round(3))

# Extra: how often does each model pick the winner? (draws excluded)
decisive = data[target] != 0
checks["decisive_matches"] = int(decisive.sum())
checks["rank_rule_accuracy"] = float((np.sign(data.loc[decisive, "rank_diff"]) == np.sign(data.loc[decisive, target])).mean())
for name in models:
    if name != "Baseline (mean)":
        checks["accuracy_" + name] = float((np.sign(predictions[name][decisive]) == np.sign(data.loc[decisive, target])).mean())
pd.Series(checks).to_csv("../outputs/results/task21_checks.csv", header=["value"])
print("Higher-ranked team wins:", round(checks["rank_rule_accuracy"], 3), "of", checks["decisive_matches"], "decisive matches")
