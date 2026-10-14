"""
Helper functions shared by step2 and step3 (kept short and simple).

  fit_ols()        fits the 8-variable linear regression and returns the coefficient table + checks
  compare_models() repeated 5-fold cross-validation for several regression models
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def fit_ols(data, features, target, cluster_ids=None):
    """Fit ordinary least squares on ALL rows (to interpret the coefficients).

    Robust standard errors are used because football goals are not perfectly 'well-behaved':
      - Task 2.1: HC3 robust errors
      - Task 2.2: errors clustered by match (the two teams in a match are not independent)
    """
    X = sm.add_constant(data[features])
    y = data[target]
    if cluster_ids is None:
        model = sm.OLS(y, X).fit(cov_type="HC3")
    else:
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cluster_ids})

    table = pd.DataFrame({
        "coefficient": model.params,
        "std_error": model.bse,
        "p_value": model.pvalues,
        "ci_low": model.conf_int()[0],
        "ci_high": model.conf_int()[1],
    })
    table["VIF"] = [np.nan] + [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]

    checks = {
        "n_rows": len(data),
        "r2": model.rsquared,
        "adjusted_r2": model.rsquared_adj,
        "max_vif": table["VIF"].max(),
        "breusch_pagan_p": het_breuschpagan(model.resid, X)[1],   # small p = uneven spread of errors
        "shapiro_p": stats.shapiro(model.resid)[1],               # small p = errors not normal
        "negative_predictions_pct": float((model.fittedvalues < 0).mean() * 100),
    }
    return table, checks


def compare_models(data, features, target, models, group_column=None, repeats=30, folds=5):
    """Repeated k-fold cross-validation.

    models: {"name": (sklearn_model, list_of_columns_to_use)}
    group_column: rows with the same value stay in the same fold (Task 2.2 uses match_id,
                  so the two teams of one match are never split between train and test).
    Returns a table with RMSE, MAE and R2 (mean and SD over the repeats), plus
    'beats_ols_pct' = how often a model had lower RMSE than the 8-variable OLS.
    """
    y = data[target].values
    groups = data[group_column].values if group_column else np.arange(len(data))
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(42)

    scores = {name: {"rmse": [], "mae": [], "r2": []} for name in models}
    predictions = {name: np.zeros(len(data)) for name in models}    # average out-of-fold prediction

    for _ in range(repeats):
        # give each group a random fold number 0..folds-1
        shuffled = rng.permutation(unique_groups)
        fold_of_group = {g: i % folds for i, g in enumerate(shuffled)}
        fold = np.array([fold_of_group[g] for g in groups])

        for name, (model, columns) in models.items():
            X = data[columns].values
            out_of_fold = np.zeros(len(data))
            for k in range(folds):
                train, test = fold != k, fold == k
                model.fit(X[train], y[train])
                out_of_fold[test] = model.predict(X[test])
            error = y - out_of_fold
            scores[name]["rmse"].append(np.sqrt(np.mean(error ** 2)))
            scores[name]["mae"].append(np.mean(np.abs(error)))
            scores[name]["r2"].append(1 - np.sum(error ** 2) / np.sum((y - y.mean()) ** 2))
            predictions[name] += out_of_fold / repeats

    ols_rmse = np.array(scores["OLS (8 variables)"]["rmse"])
    rows = []
    for name, s in scores.items():
        rows.append({
            "model": name,
            "RMSE": np.mean(s["rmse"]), "RMSE_sd": np.std(s["rmse"]),
            "MAE": np.mean(s["mae"]), "R2": np.mean(s["r2"]),
            "beats_ols_pct": 100 * np.mean(np.array(s["rmse"]) < ols_rmse),
        })
    return pd.DataFrame(rows), predictions
