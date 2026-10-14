"""
STEP 3 - Task 2.2: predict the GOALS SCORED by one team in one match.

208 rows (two per match), 8 explanatory variables known before kick-off.
Same structure as step 2.  Rows from the same match always stay together when we
cross-validate (the two teams in a match are linked), and a Poisson model is added
because goals are counts.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, PoissonRegressor, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model_helpers import compare_models, fit_ols

data = pd.read_csv("../data/team_matches_208.csv")
# 4 variables shared with Task 2.1 (the brief allows at most 4) + 4 used only here
features = ["rank_advantage", "is_host", "same_confederation", "stage_num",
            "own_gf_form", "opp_ga_form", "team_is_european", "opp_is_european"]
target = "goals_scored"

# ---- Part A: the 8-variable linear regression ----
table, checks = fit_ols(data, features, target, cluster_ids=data["match_id"])
table.to_csv("../outputs/results/task22_ols_coefficients.csv")
print(table.round(3))
print({k: round(float(v), 3) for k, v in checks.items()})
correlations = data[features].corr().abs().values - np.eye(len(features))
checks["max_correlation_between_predictors"] = correlations.max()
pd.Series(checks).to_csv("../outputs/results/task22_checks.csv", header=["value"])

# ---- Part B: compare models with repeated 5-fold cross-validation ----
models = {
    "Baseline (mean)":   (DummyRegressor(), features),
    "OLS (rank only)":   (LinearRegression(), ["rank_advantage"]),
    "OLS (8 variables)": (LinearRegression(), features),
    "Ridge":             (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), features),
    "Lasso":             (make_pipeline(StandardScaler(), LassoCV(cv=5)), features),
    "Poisson":           (make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0)), features),
    "Random forest":     (RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=0), features),
    "Gradient boosting": (GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0), features),
}
comparison, predictions = compare_models(data, features, target, models, group_column="match_id")
comparison.to_csv("../outputs/results/task22_model_comparison.csv", index=False)
print(comparison.round(3))
