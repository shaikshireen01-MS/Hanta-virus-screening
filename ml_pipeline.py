"""
ml_pipeline.py  –  Ensemble ML training, LOO-CV, uncertainty quantification,
                    ADMET filtering, and virtual screening.

Pipeline:
  1. LOO-CV on training set  →  per-model R², RMSE, residual std
  2. Fit final models on full training set
  3. Screen test library     →  RF prediction, GB prediction, ensemble mean
  4. Uncertainty metrics     →  model disagreement, 95 % CI, confidence score
  5. Lipinski Ro5 filter     →  flag ADMET pass / fail per compound
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error

from data import (
    FEATURE_NAMES,
    TRAINING_COMPOUNDS, X_TRAIN, Y_TRAIN,
    TEST_COMPOUNDS, X_TEST,
    LIPINSKI, RF_PARAMS, GB_PARAMS,
    RESULTS_CSV,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def lipinski_pass(row: np.ndarray) -> bool:
    """Return True if compound passes all Lipinski Ro5 criteria."""
    mw, logp, hbd, hba, rot = row[0], row[1], row[2], row[3], row[4]
    return (
        mw   <= LIPINSKI["MW"]
        and logp  <= LIPINSKI["LogP"]
        and hbd   <= LIPINSKI["HBD"]
        and hba   <= LIPINSKI["HBA"]
        and rot    < LIPINSKI["RotBonds"]
    )


def confidence_score(total_uncertainty: float) -> float:
    """Confidence score (%) = 100 × exp(−σ_total)."""
    return float(100.0 * np.exp(-total_uncertainty))


# ─── Step 1 · Leave-One-Out Cross-Validation ──────────────────────────────────

def run_loo_cv(X, y, rf_params=RF_PARAMS, gb_params=GB_PARAMS, verbose=True):
    """
    Run LOO-CV on the training set.

    Returns
    -------
    dict with keys:
        rf_loo_preds, gb_loo_preds, ens_loo_preds  (np.ndarray, len N)
        rf_r2, gb_r2, ens_r2                        (float)
        rf_rmse, gb_rmse, ens_rmse                  (float)
        residual_std, ci_95                          (float)
    """
    loo = LeaveOneOut()
    rf_preds, gb_preds = np.zeros(len(y)), np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        rf = RandomForestRegressor(**rf_params)
        gb = GradientBoostingRegressor(**gb_params)
        rf.fit(X[train_idx], y[train_idx])
        gb.fit(X[train_idx], y[train_idx])
        rf_preds[fold] = rf.predict(X[test_idx])[0]
        gb_preds[fold] = gb.predict(X[test_idx])[0]

    ens_preds = (rf_preds + gb_preds) / 2.0
    residuals  = y - ens_preds
    res_std    = float(residuals.std())

    results = {
        "rf_loo_preds":  rf_preds,
        "gb_loo_preds":  gb_preds,
        "ens_loo_preds": ens_preds,
        "rf_r2":         r2_score(y, rf_preds),
        "gb_r2":         r2_score(y, gb_preds),
        "ens_r2":        r2_score(y, ens_preds),
        "rf_rmse":       rmse(y, rf_preds),
        "gb_rmse":       rmse(y, gb_preds),
        "ens_rmse":      rmse(y, ens_preds),
        "residual_std":  res_std,
        "ci_95":         1.96 * res_std,
        # Per-fold R² for Figure 3 CV panel (approximate 3-fold summary)
        "fold_r2": [
            r2_score(y[:4],  ens_preds[:4]),
            r2_score(y[4:8], ens_preds[4:8]),
            r2_score(y[8:],  ens_preds[8:]),
        ],
    }

    if verbose:
        print("\n─── LOO Cross-Validation ───────────────────────────────────")
        print(f"  RF   R²={results['rf_r2']:+.3f}   RMSE={results['rf_rmse']:.3f} kcal/mol")
        print(f"  GB   R²={results['gb_r2']:+.3f}   RMSE={results['gb_rmse']:.3f} kcal/mol")
        print(f"  ENS  R²={results['ens_r2']:+.3f}   RMSE={results['ens_rmse']:.3f} kcal/mol")
        print(f"  Residual std = {res_std:.3f}  |  95% CI = ±{results['ci_95']:.3f} kcal/mol")

    return results


# ─── Step 2 · Fit final models on full training set ───────────────────────────

def fit_final_models(X, y, rf_params=RF_PARAMS, gb_params=GB_PARAMS):
    """Fit RF and GB on the complete training set. Also extract feature importances."""
    rf = RandomForestRegressor(**rf_params)
    gb = GradientBoostingRegressor(**gb_params)
    rf.fit(X, y)
    gb.fit(X, y)
    # Average feature importances from both models
    avg_imp = (rf.feature_importances_ + gb.feature_importances_) / 2.0
    return rf, gb, avg_imp


# ─── Step 3+4 · Screen test library with uncertainty ─────────────────────────

def screen_compounds(rf_model, gb_model, X_test, compound_names,
                     residual_std: float, ci_95: float) -> pd.DataFrame:
    """
    Predict binding affinities for test compounds and compute uncertainty metrics.

    Returns a DataFrame with one row per compound.
    """
    rf_pred  = rf_model.predict(X_test)
    gb_pred  = gb_model.predict(X_test)
    ens_pred = (rf_pred + gb_pred) / 2.0

    disagreement    = np.abs(rf_pred - gb_pred)
    total_unc       = np.sqrt(disagreement**2 + residual_std**2)
    conf_scores     = np.array([confidence_score(u) for u in total_unc])
    admet_pass      = np.array([lipinski_pass(X_test[i]) for i in range(len(compound_names))])

    df = pd.DataFrame({
        "Compound":            compound_names,
        "RF_Affinity":         np.round(rf_pred,  3),
        "GB_Affinity":         np.round(gb_pred,  3),
        "Ensemble_Affinity":   np.round(ens_pred, 3),
        "Disagreement":        np.round(disagreement, 3),
        "CI_95":               np.round([ci_95] * len(compound_names), 3),
        "Total_Uncertainty":   np.round(total_unc, 3),
        "Confidence_pct":      np.round(conf_scores, 1),
        "ADMET_Pass":          admet_pass,
        # Descriptor columns for ADMET figure
        "MW":                  X_test[:, 0],
        "LogP":                X_test[:, 1],
        "HBD":                 X_test[:, 2].astype(int),
        "HBA":                 X_test[:, 3].astype(int),
        "RotBonds":            X_test[:, 4].astype(int),
        "TPSA":                X_test[:, 5],
    })

    # Sort approved compounds by descending affinity
    approved = (
        df[df["ADMET_Pass"]]
        .sort_values("Ensemble_Affinity", ascending=False)
        .reset_index(drop=True)
    )
    approved.index += 1  # 1-based rank

    print("\n─── Virtual Screening Results (ADMET-approved) ─────────────────")
    print(approved[["Compound", "Ensemble_Affinity", "Confidence_pct", "CI_95"]].to_string())
    print(f"\n  Total screened: {len(df)}  |  Passed ADMET: {approved.shape[0]}")

    return df, approved


# ─── Step 5 · Save CSV results ────────────────────────────────────────────────

def save_results(df_all: pd.DataFrame, approved: pd.DataFrame, cv_results: dict):
    """Save screening results to CSV."""
    approved_out = approved[
        ["Compound", "Ensemble_Affinity", "RF_Affinity", "GB_Affinity",
         "Confidence_pct", "CI_95", "MW", "LogP", "HBD", "HBA"]
    ].copy()
    approved_out.index.name = "Rank"
    approved_out.to_csv(RESULTS_CSV)
    print(f"\n  Results saved → {RESULTS_CSV}")


# ─── Master runner ────────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 62)
    print("  HANTAVIRUS ANTIVIRAL ML PIPELINE")
    print("=" * 62)

    # 1. LOO-CV
    cv = run_loo_cv(X_TRAIN, Y_TRAIN)

    # 2. Final models
    rf_model, gb_model, feature_imp = fit_final_models(X_TRAIN, Y_TRAIN)
    print(f"\n─── Feature Importances (ensemble average) ──────────────────")
    for name, imp in sorted(zip(FEATURE_NAMES, feature_imp), key=lambda x: x[1]):
        print(f"  {name:<20} {imp:.4f}")

    # 3+4. Screen test library
    df_all, approved = screen_compounds(
        rf_model, gb_model, X_TEST, TEST_COMPOUNDS,
        residual_std=cv["residual_std"], ci_95=cv["ci_95"],
    )

    # 5. Save
    save_results(df_all, approved, cv)

    return cv, rf_model, gb_model, feature_imp, df_all, approved


if __name__ == "__main__":
    run_pipeline()
