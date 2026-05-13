"""
figures.py  –  Generate all three publication-quality manuscript figures.

Figure 1 · Top Hantavirus Antiviral Candidates
    Horizontal bar chart, predicted binding affinity |ΔG| (kcal/mol)

Figure 2 · ADMET Property Analysis
    Four-panel: MW histogram, LogP histogram, HBD vs HBA scatter, RotBonds histogram

Figure 3 · Model Performance and Uncertainty
    Left panel:  Ensemble feature importances (horizontal bar)
    Right panel: LOO-CV R² per fold with mean reference line
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from data import FEATURE_NAMES, FIGURE_PATHS


# ─── Style constants ──────────────────────────────────────────────────────────
BG      = "#EAECF0"
C_RED   = "#E05C4B"
C_BLUE  = "#4A90D9"
C_GREEN = "#4CAF7D"
C_PURP  = "#8E6BBF"
FONT    = "DejaVu Sans"

BASE_STYLE = {
    "font.family":       FONT,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.facecolor":    BG,
    "figure.facecolor":  "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.0,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
}


# ─── Figure 1 · Top Hantavirus Antiviral Candidates ──────────────────────────

def plot_figure1(approved_df, save_path: str):
    """
    Horizontal bar chart of predicted |ΔG| for ADMET-approved compounds,
    sorted descending.  Same style as original manuscript figure.
    """
    with plt.rc_context(BASE_STYLE):
        compounds  = approved_df["Compound"].tolist()
        affinities = approved_df["Ensemble_Affinity"].tolist()

        # Reverse so highest is at top
        compounds  = compounds[::-1]
        affinities = affinities[::-1]

        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        ax.set_facecolor(BG)

        bars = ax.barh(
            compounds, affinities,
            color=C_RED, edgecolor="#222222", linewidth=0.8, height=0.65,
        )

        # Value labels at bar ends
        for bar, val in zip(bars, affinities):
            ax.text(
                val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=9.5, color="#333333",
            )

        ax.set_xlabel("Affinity |ΔG| (kcal/mol)", fontsize=11)
        ax.set_title("Top Hantavirus Antivirals", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlim(0, max(affinities) * 1.15)
        ax.tick_params(axis="y", labelsize=10)

        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Figure 1 saved → {save_path}")


# ─── Figure 2 · ADMET Property Analysis ──────────────────────────────────────

def plot_figure2(approved_df, save_path: str):
    """
    2 × 2 panel ADMET analysis:
      [0,0] MW histogram         [0,1] LogP histogram
      [1,0] HBD vs HBA scatter   [1,1] RotBonds histogram
    """
    mw  = approved_df["MW"].values.astype(float)
    lp  = approved_df["LogP"].values.astype(float)
    hbd = approved_df["HBD"].values.astype(int)
    hba = approved_df["HBA"].values.astype(int)
    rot = approved_df["RotBonds"].values.astype(int)

    with plt.rc_context(BASE_STYLE):
        fig = plt.figure(figsize=(11, 8))
        gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

        # ── [0,0] Molecular Weight ─────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0])
        bins_mw = np.arange(mw.min() - 15, mw.max() + 40, 30)
        ax0.hist(mw, bins=bins_mw, color=C_BLUE, edgecolor="white", linewidth=0.8)
        ax0.axvline(500, color="#CC0000", linestyle="--", linewidth=1.6,
                    label="Ro5 limit (500 Da)")
        ax0.set_xlabel("MW (Da)")
        ax0.set_ylabel("Count")
        ax0.set_title("Molecular Weight")
        ax0.legend(fontsize=9)

        # ── [0,1] LogP ────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1])
        bins_lp = np.arange(np.floor(lp.min()) - 0.5, np.ceil(lp.max()) + 1.5, 1)
        ax1.hist(lp, bins=bins_lp, color=C_RED, edgecolor="white", linewidth=0.8)
        ax1.axvline(5, color="#CC0000", linestyle="--", linewidth=1.6,
                    label="Ro5 limit (5)")
        ax1.set_xlabel("LogP")
        ax1.set_ylabel("Count")
        ax1.set_title("Lipophilicity")
        ax1.legend(fontsize=9)

        # ── [1,0] HBD vs HBA scatter ──────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        scatter = ax2.scatter(
            hbd, hba, s=90, color=C_GREEN,
            edgecolors="#2E7D32", linewidths=0.8, alpha=0.85,
        )
        ax2.set_xlabel("H-bond Donors")
        ax2.set_ylabel("H-bond Acceptors")
        ax2.set_title("H-bond Profile")
        ax2.set_xticks(range(int(hbd.min()), int(hbd.max()) + 2))
        ax2.set_yticks(range(int(hba.min()), int(hba.max()) + 2))

        # ── [1,1] Rotatable Bonds ─────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        bins_rot = np.arange(-0.5, rot.max() + 1.5, 1)
        ax3.hist(rot, bins=bins_rot, color=C_PURP, edgecolor="white", linewidth=0.8)
        ax3.axvline(7.5, color="#CC0000", linestyle="--", linewidth=1.6,
                    label="Ro5 limit (<8)")
        ax3.set_xlabel("Rotatable Bonds")
        ax3.set_ylabel("Count")
        ax3.set_title("Flexibility")
        ax3.legend(fontsize=9)

        fig.suptitle("ADMET Analysis", fontsize=15, fontweight="bold", y=1.01)
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Figure 2 saved → {save_path}")


# ─── Figure 3 · Model Performance and Uncertainty ────────────────────────────

def plot_figure3(cv_results: dict, feature_importances: np.ndarray, save_path: str):
    """
    Left:  Horizontal bar chart of ensemble-averaged feature importances.
    Right: LOO-CV fold R² with mean reference line.
    """
    # Sort features by importance
    pairs     = sorted(zip(FEATURE_NAMES, feature_importances), key=lambda x: x[1])
    feat_lbls = [p[0] for p in pairs]
    feat_vals = [p[1] for p in pairs]

    fold_r2   = cv_results["fold_r2"]   # list of 3 floats
    mean_r2   = float(np.mean(fold_r2))

    with plt.rc_context(BASE_STYLE):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

        # ── Left: Feature Importances ─────────────────────────────────────
        ax_l.set_facecolor(BG)
        ax_l.barh(
            feat_lbls, feat_vals,
            color=C_BLUE, edgecolor="#222222", linewidth=0.6, height=0.6,
        )
        ax_l.set_xlabel("Importance")
        ax_l.set_title("Feature Importance")
        ax_l.set_xlim(0, max(feat_vals) * 1.25)
        for val, y_pos in zip(feat_vals, range(len(feat_vals))):
            ax_l.text(
                val + max(feat_vals) * 0.02, y_pos,
                f"{val:.3f}", va="center", fontsize=9, color="#333333",
            )

        # ── Right: CV Performance per fold ────────────────────────────────
        ax_r.set_facecolor(BG)
        x_folds = [1, 2, 3]
        ax_r.plot(
            x_folds, fold_r2, "o-",
            color=C_GREEN, linewidth=2, markersize=8,
            markerfacecolor=C_GREEN, markeredgecolor="#1B5E20", markeredgewidth=1,
        )
        ax_r.axhline(
            mean_r2, color="#CC0000", linestyle="--", linewidth=1.8,
            label=f"Mean R² = {mean_r2:.2f}",
        )
        ax_r.set_xlabel("Fold")
        ax_r.set_ylabel("R²")
        ax_r.set_title("CV Performance")
        ax_r.set_xticks(x_folds)
        ax_r.legend(fontsize=10)

        fig.suptitle(
            "Model Performance and Uncertainty",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Figure 3 saved → {save_path}")


# ─── Master figure generator ─────────────────────────────────────────────────

def generate_all_figures(approved_df, cv_results: dict, feature_importances: np.ndarray):
    """Generate and save all three figures. Returns dict of paths."""
    print("\n─── Generating Figures ─────────────────────────────────────────")
    plot_figure1(approved_df,  FIGURE_PATHS["fig1_antivirals"])
    plot_figure2(approved_df,  FIGURE_PATHS["fig2_admet"])
    plot_figure3(cv_results, feature_importances, FIGURE_PATHS["fig3_model_perf"])
    return FIGURE_PATHS


if __name__ == "__main__":
    # Quick smoke-test with dummy data
    import pandas as pd
    dummy = pd.DataFrame({
        "Compound":          ["Remdesivir", "Favipiravir", "Ribavirin"],
        "Ensemble_Affinity": [8.03, 7.20, 6.69],
        "MW":   [368, 253, 155],
        "LogP": [2.85, 1.05, -1.84],
        "HBD":  [3, 1, 3],
        "HBA":  [8, 5, 3],
        "RotBonds": [6, 2, 1],
    })
    cv_dummy = {
        "fold_r2": [0.81, 0.83, 0.82],
        "residual_std": 0.955,
        "ci_95": 1.871,
    }
    imp_dummy = np.array([0.05, 0.10, 0.08, 0.07, 0.06, 0.12, 0.08, 0.04, 0.11])
    imp_dummy /= imp_dummy.sum()
    generate_all_figures(dummy, cv_dummy, imp_dummy)
    print("Smoke-test passed.")
