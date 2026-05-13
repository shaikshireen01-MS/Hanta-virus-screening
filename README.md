# Hantavirus Antiviral ML Pipeline

**Machine Learning-Driven Virtual Screening with Uncertainty Quantification for Hantavirus Antiviral Candidates**

> Sk. Shireen¹ · Dr. Sk. Md Nayeem² · Sk. Md Rameez Arhan³
>
> ¹ University of Europe for Applied Sciences, Potsdam, Germany  
> ² Dept. of Physics, Government Degree College for Women, Guntur, AP, India  
> ³ Dept. of Mechanical Engineering, NIT Warangal, Telangana, India

---

## Overview

This repository contains the **complete, reproducible Python pipeline** for the computational study:

> *"Machine Learning-Driven Virtual Screening with Uncertainty Quantification Identifies Promising Hantavirus Antiviral Candidates: A Preliminary Computational Study"*

Running `python run_pipeline.py` from the repo root will:

1. **Train** an ensemble of Random Forest + Gradient Boosting models on 12 known antivirals  
2. **Evaluate** via Leave-One-Out cross-validation with honest R² and RMSE reporting  
3. **Screen** 13 test compounds with per-compound uncertainty quantification  
4. **Filter** candidates by Lipinski's Rule of 5 (ADMET)  
5. **Generate** 3 publication-quality figures  
6. **Assemble** the complete Word manuscript (`.docx`) with embedded figures and tables  

Total runtime: **< 2 minutes** on a standard laptop.

---

## Repository Structure

```
hantavirus-antiviral-ml/
│
├── run_pipeline.py          # ← Master entry point: run this
│
├── src/
│   ├── data.py              # Training data, test library, feature definitions
│   ├── ml_pipeline.py       # LOO-CV, ensemble training, screening, uncertainty
│   ├── figures.py           # All 3 publication figures (matplotlib)
│   └── build_manuscript.py  # Word document assembly (python-docx)
│
├── figures/                 # Auto-generated figures (created on first run)
│   ├── fig1_top_antivirals.png
│   ├── fig2_admet_analysis.png
│   └── fig3_model_performance.png
│
├── outputs/                 # Auto-generated outputs (created on first run)
│   ├── Hantavirus_Antiviral_Manuscript_FINAL.docx
│   └── screening_results.csv
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/[username]/hantavirus-antiviral-ml.git
cd hantavirus-antiviral-ml
```

### 2. Set up the environment

**Option A — pip (recommended)**

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — conda**

```bash
conda create -n hanta python=3.11
conda activate hanta
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python run_pipeline.py
```

---

## Requirements

| Package        | Minimum version | Purpose                        |
|----------------|-----------------|--------------------------------|
| Python         | 3.9             |                                |
| numpy          | 1.24.0          | Array operations               |
| pandas         | 2.0.0           | Data frames, CSV export        |
| scikit-learn   | 1.3.0           | RF, GB, LOO-CV, metrics        |
| matplotlib     | 3.7.0           | All 3 figures                  |
| scipy          | 1.11.0          | Statistical utilities          |
| python-docx    | 1.1.0           | Word manuscript assembly       |
| Pillow         | 10.0.0          | Image handling in docx         |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Ubuntu / Debian setup (from scratch)

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git

git clone https://github.com/[username]/hantavirus-antiviral-ml.git
cd hantavirus-antiviral-ml

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py
```

---

## Outputs

After a successful run you will find:

| File | Description |
|------|-------------|
| `outputs/Hantavirus_Antiviral_Manuscript_FINAL.docx` | Complete manuscript with figures and tables |
| `outputs/screening_results.csv` | Ranked virtual screening results |
| `figures/fig1_top_antivirals.png` | Horizontal bar chart — predicted binding affinities |
| `figures/fig2_admet_analysis.png` | 4-panel ADMET property distributions |
| `figures/fig3_model_performance.png` | Feature importance + CV performance |

---

## Methods Summary

### Training data
12 known broad-spectrum antivirals with consensus binding affinities (kcal/mol) from published crystallographic and biophysical studies on viral RNA polymerases.

### Features (9 physicochemical descriptors per compound)
Molecular Weight, LogP, H-bond Donors, H-bond Acceptors, Rotatable Bonds, TPSA, Atom Count, Aromatic Rings, Molar Refractivity.

### Models
- **Random Forest**: 150 trees, max_depth=3, min_samples_split=3  
- **Gradient Boosting**: 150 estimators, learning_rate=0.05, max_depth=2  
- **Ensemble**: Equal-weight average of RF and GB predictions

### Cross-validation
Leave-One-Out CV (LOO-CV) on N=12. Negative R² values are **expected and acceptable** for LOO-CV on extremely small datasets; RMSE is the primary metric.

### Uncertainty quantification
For each test compound:
```
Disagreement    = |RF_pred − GB_pred|
Total_Unc       = sqrt(Disagreement² + Residual_Std²)
Confidence (%)  = 100 × exp(−Total_Unc)
95% CI          = ±1.96 × Residual_Std
```

### ADMET filtering
Lipinski's Rule of 5: MW ≤ 500 Da, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10, RotBonds < 8.

---

## Key Results

| Rank | Compound             | Affinity (kcal/mol) | Confidence (%) |
|------|----------------------|---------------------|----------------|
| 1    | Remdesivir           | 8.03                | 27.6           |
| 2    | Favipiravir          | 7.20                | 38.9           |
| 3    | Ribavirin            | 6.69                | 38.4           |
| 4    | Heteroaromatic_G     | 6.48                | 38.3           |
| 5    | Modified_Nucleoside_F| 6.33                | 38.8           |

Ensemble LOO-CV: RMSE = 0.943 kcal/mol · 95% CI = ±1.85 kcal/mol

---

## Notes on Reproducibility

- All random states are fixed (`random_state=42`) for exact reproducibility across machines.
- Predicted affinity values may differ slightly (±0.01–0.05 kcal/mol) from the manuscript due to floating-point differences between scikit-learn versions. This does not affect compound ranking or conclusions.
- If you have RDKit installed, you can replace the hand-curated descriptor table in `src/data.py` with RDKit-computed values using actual SMILES strings for improved accuracy.

---

## Citation

If you use this code, please cite:

```
Shireen S, Nayeem SKM, Arhan SKMR. (2025).
Machine Learning-Driven Virtual Screening with Uncertainty Quantification
Identifies Promising Hantavirus Antiviral Candidates: A Preliminary Computational Study.
[Journal Name]. DOI: [pending]
```

---

## License

MIT License — see `LICENSE` for details.
