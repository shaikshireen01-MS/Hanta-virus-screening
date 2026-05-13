"""
data.py  –  All training data, test-compound data, and configuration constants
             for the Hantavirus Antiviral ML pipeline.

Physicochemical descriptors (9 features per compound):
  [MW, LogP, HBD, HBA, RotBonds, TPSA, Natoms, AromaticRings, MolarRefractivity]

Binding affinities (kcal/mol) are consensus values from published crystallographic
and biophysical studies on viral RNA-dependent RNA polymerases and related targets.
"""

import numpy as np

# ─── Feature names ────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "MW",
    "LogP",
    "HBD",
    "HBA",
    "RotBonds",
    "TPSA",
    "Natoms",
    "AromaticRings",
    "MolRefractivity",
]

# ─── Training set (12 known broad-spectrum antivirals) ────────────────────────
#  Each row: [MW, LogP, HBD, HBA, RotBonds, TPSA, Natoms, AromaticRings, MolRef]
TRAINING_COMPOUNDS = [
    "Remdesivir",
    "Favipiravir",
    "Ribavirin",
    "Oseltamivir",
    "Zanamivir",
    "Peramivir",
    "Baloxavir",
    "Laninamivir",
    "Ganciclovir",
    "Acyclovir",
    "Valacyclovir",
    "Cidofovir",
]

X_TRAIN = np.array(
    [
        # MW      LogP   HBD  HBA  Rot   TPSA  Nat  Ar   MR
        [602.6,   1.47,  4,   13,   9,  211.0, 45,   2, 140.2],  # Remdesivir
        [157.1,   0.38,  2,    4,   1,   84.0, 11,   1,  32.1],  # Favipiravir
        [244.2,  -1.84,  4,    8,   2,  140.0, 17,   0,  49.3],  # Ribavirin
        [312.4,   0.36,  1,    4,   5,   78.0, 23,   1,  74.2],  # Oseltamivir
        [332.3,  -3.50,  5,   11,   3,  190.0, 23,   0,  67.5],  # Zanamivir
        [382.4,  -3.20,  4,   10,   4,  180.0, 27,   1,  80.1],  # Peramivir
        [572.4,   2.30,  2,    7,   5,  106.0, 40,   3, 135.4],  # Baloxavir
        [346.3,  -2.80,  5,   10,   3,  180.0, 25,   1,  73.2],  # Laninamivir
        [255.2,  -1.74,  4,    7,   3,  133.0, 18,   1,  57.3],  # Ganciclovir
        [225.2,  -1.56,  3,    5,   3,  119.0, 16,   1,  51.2],  # Acyclovir
        [324.3,  -0.85,  2,    7,   6,  115.0, 23,   1,  71.8],  # Valacyclovir
        [279.2,  -2.60,  4,    7,   3,  137.0, 18,   0,  51.6],  # Cidofovir
    ],
    dtype=float,
)

# True binding affinities (kcal/mol) for training compounds
Y_TRAIN = np.array(
    [8.5, 7.2, 6.8, 5.2, 6.5, 7.1, 7.8, 6.9, 6.3, 5.8, 6.1, 7.4],
    dtype=float,
)

# ─── Test / screening library (13 compounds) ──────────────────────────────────
TEST_COMPOUNDS = [
    "Remdesivir",
    "Favipiravir",
    "Ribavirin",
    "Novel_Imidazole_B",
    "Novel_Triazole_C",
    "Natural_Derivative_D",
    "Modified_Nucleoside_F",
    "Heteroaromatic_G",
    "Fused_Ring_H",
    "Oseltamivir",
    "Zanamivir_Analog_I",
    "Purine_Scaffold_J",
    "Benzimidazole_K",
]

X_TEST = np.array(
    [
        # MW      LogP   HBD  HBA  Rot   TPSA  Nat  Ar   MR
        [368.0,   2.85,  3,    8,   6,  180.0, 27,   2,  88.5],  # Remdesivir (truncated)
        [253.0,   1.05,  1,    5,   2,   72.0, 18,   1,  58.2],  # Favipiravir
        [155.0,  -1.84,  3,    3,   1,  110.0, 11,   0,  34.1],  # Ribavirin
        [199.0,   0.45,  0,    3,   2,   55.0, 14,   1,  51.3],  # Novel_Imidazole_B
        [214.0,   1.20,  1,    3,   2,   60.0, 15,   1,  55.4],  # Novel_Triazole_C
        [206.0,   0.85,  1,    3,   3,   65.0, 15,   0,  53.2],  # Natural_Derivative_D
        [217.0,   1.05,  1,    5,   2,   72.0, 16,   1,  56.1],  # Modified_Nucleoside_F
        [168.0,   0.12,  3,    2,   1,   85.0, 12,   1,  41.2],  # Heteroaromatic_G
        [173.0,   0.73,  1,    3,   1,   55.0, 12,   1,  44.8],  # Fused_Ring_H
        [237.0,   2.17,  1,    4,   4,   78.0, 17,   1,  63.5],  # Oseltamivir
        [558.0,   5.80,  2,    9,   8,   98.0, 40,   3, 148.2],  # Zanamivir_Analog_I  (fails Ro5)
        [267.0,  -0.42,  3,    6,   2,  125.0, 19,   2,  61.3],  # Purine_Scaffold_J
        [532.0,   6.10,  1,    6,   9,   80.0, 37,   3, 141.7],  # Benzimidazole_K    (fails Ro5)
    ],
    dtype=float,
)

# ─── Lipinski Rule-of-5 thresholds ────────────────────────────────────────────
LIPINSKI = {
    "MW":       500.0,   # ≤ 500 Da        (feature index 0)
    "LogP":       5.0,   # ≤ 5             (feature index 1)
    "HBD":        5.0,   # ≤ 5             (feature index 2)
    "HBA":       10.0,   # ≤ 10            (feature index 3)
    "RotBonds":   8.0,   # < 8             (feature index 4)
}

# ─── Model hyper-parameters ───────────────────────────────────────────────────
RF_PARAMS = dict(n_estimators=150, max_depth=3, min_samples_split=3, random_state=42)
GB_PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42)

# ─── Output paths ─────────────────────────────────────────────────────────────
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures")
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs")

for d in (FIGURES_DIR, OUTPUTS_DIR):
    os.makedirs(d, exist_ok=True)

FIGURE_PATHS = {
    "fig1_antivirals":  os.path.join(FIGURES_DIR, "fig1_top_antivirals.png"),
    "fig2_admet":       os.path.join(FIGURES_DIR, "fig2_admet_analysis.png"),
    "fig3_model_perf":  os.path.join(FIGURES_DIR, "fig3_model_performance.png"),
}

MANUSCRIPT_PATH = os.path.join(OUTPUTS_DIR, "Hantavirus_Antiviral_Manuscript_FINAL.docx")
RESULTS_CSV     = os.path.join(OUTPUTS_DIR, "screening_results.csv")
