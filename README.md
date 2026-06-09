# 🦠 Hantavirus Antiviral ML Pipeline

> Machine Learning-Driven Virtual Screening with Uncertainty Quantification Identifies Promising Hantavirus Antiviral Candidates: A Preliminary Computational Study

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Methodology](#methodology)
4. [Project Structure](#project-structure)
5. [Models Used](#models-used)
6. [Results](#results)
7. [How to Run](#how-to-run)
8. [Publication](#publication)
9. [Author](#author)

---

## 🎯 Project Overview

This project applies machine learning-driven virtual screening combined with uncertainty quantification to identify promising antiviral drug candidates against Hantavirus. The pipeline integrates cheminformatics (RDKit), molecular fingerprinting, and ensemble ML models to screen large compound libraries and rank candidates by predicted bioactivity and confidence.

---

## ❓ Problem Statement

Hantavirus causes severe pulmonary and renal syndromes with high mortality rates. No FDA-approved antivirals currently exist. Traditional drug discovery is expensive and slow. This project uses computational ML screening to prioritize candidate compounds for experimental validation — dramatically reducing time and cost.

---

## 🔬 Methodology

1. **Data Collection** — Bioactivity data from ChEMBL database for Hantavirus targets
2. **Molecular Featurization** — Morgan fingerprints + RDKit descriptors
3. **Model Training** — Multiple ML classifiers with cross-validation
4. **Uncertainty Quantification** — Confidence scoring for predictions
5. **Virtual Screening** — Ranking compound library by predicted activity
6. **Hit Identification** — Top candidates selected for further analysis

---

## 📁 Project Structure

```
Hanta-virus-screening/
│
├── hantavirus_ml_pipeline.py      # Main ML pipeline
├── ml_pipeline.py                 # Pipeline modules
├── data.py                        # Data loading & preprocessing
├── figures.py                     # Visualization functions
├── build_manuscript.py            # Manuscript figure builder
├── hantavirus_ubuntu_standalone.py # Standalone runner
├── requirements.txt               # Dependencies
├── Hantavirus_Cited_Manuscript.docx  # Research manuscript
├── Explanation-what done.doc      # Project explanation
├── Hantavirus_EDA_Modeling.ipynb  # EDA + modeling notebook
└── README.md
```

---

## 🤖 Models Used

| Model | Purpose |
|---|---|
| Random Forest | Primary classifier — bioactivity prediction |
| Gradient Boosting | Ensemble prediction |
| Support Vector Machine | Margin-based classification |
| Logistic Regression | Baseline model |
| Conformal Prediction | Uncertainty quantification |

---

## 📈 Results

- Successfully screened compound libraries for Hantavirus antiviral activity
- Identified top candidate compounds with high predicted bioactivity
- Uncertainty quantification provides confidence intervals for each prediction
- Results validated against known antiviral compounds in literature

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/shaikshireen01-MS/Hanta-virus-screening.git
cd Hanta-virus-screening
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the main pipeline
```bash
python hantavirus_ml_pipeline.py
```

### 4. Run standalone version
```bash
python hantavirus_ubuntu_standalone.py
```

### 5. View notebook
```bash
jupyter notebook Hantavirus_EDA_Modeling.ipynb
```

---

## 📄 Publication

This work is associated with the manuscript:

**"Machine Learning-Driven Virtual Screening with Uncertainty Quantification Identifies Promising Hantavirus Antiviral Candidates: A Preliminary Computational Study"**

---

## 👩‍💻 Author

**Shaik Shireen**
M.Sc. Data Science — University of Europe for Applied Sciences, Potsdam
[GitHub](https://github.com/shaikshireen01-MS) | [ORCID](https://orcid.org/0009-0000-0438-1240)

---

## 📄 License

This project is licensed under the MIT License.
