#!/usr/bin/env python3
"""
run_pipeline.py  –  Master entry point for the Hantavirus Antiviral ML Pipeline.

Usage
-----
    cd src
    python run_pipeline.py

    # Or from repo root:
    python src/run_pipeline.py

Outputs (written to outputs/)
------------------------------
    Hantavirus_Antiviral_Manuscript_FINAL.docx   ← full manuscript with figures
    screening_results.csv                         ← ranked virtual screening table

Figures (written to figures/)
------------------------------
    fig1_top_antivirals.png
    fig2_admet_analysis.png
    fig3_model_performance.png

Requirements
------------
    pip install -r requirements.txt
    Python ≥ 3.9
"""

import sys
import os
import time

# Make src/ importable when called from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

# ── Import pipeline modules ────────────────────────────────────────────────────
try:
    from src.ml_pipeline      import run_pipeline  as run_ml
    from src.figures          import generate_all_figures
    from src.build_manuscript import build_manuscript
except ImportError:
    # Running from inside src/
    from ml_pipeline      import run_pipeline  as run_ml
    from figures          import generate_all_figures
    from build_manuscript import build_manuscript


def main():
    t0 = time.time()
    print("=" * 62)
    print("  HANTAVIRUS ANTIVIRAL VIRTUAL SCREENING PIPELINE")
    print("  Ensemble ML + Uncertainty Quantification")
    print("=" * 62)

    # ── Step 1+2+3: ML calculations ──────────────────────────────────────────
    print("\n[1/3]  Running ML pipeline …")
    cv_results, rf_model, gb_model, feature_imp, df_all, approved_df = run_ml()

    # ── Step 4: Generate figures ──────────────────────────────────────────────
    print("\n[2/3]  Generating figures …")
    generate_all_figures(approved_df, cv_results, feature_imp)

    # ── Step 5: Build Word manuscript ─────────────────────────────────────────
    print("\n[3/3]  Building manuscript …")
    manuscript_path = build_manuscript(cv_results, approved_df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 62}")
    print(f"  ✓  Pipeline complete in {elapsed:.1f} s")
    print(f"  ✓  Manuscript → {manuscript_path}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
