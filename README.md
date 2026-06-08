# CASTLE: Competency-Aware Stacked Tutoring and Learner Explainability

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Institution](https://img.shields.io/badge/Institution-KNUST-gold)

> Design of an Explainable Stacked Ensemble Learning Framework for Adaptive Digital Safety Competency Prediction in Smart Learning Environments

---

## Overview

CASTLE is a novel seven-layer machine learning framework that predicts learner digital safety competency tiers and delivers personalised, explainable adaptive instruction. It integrates SHAP-enriched stacked ensemble learning with a closed-loop adaptive content recommendation system, bridging the gap between explainable AI, ensemble learning, and adaptive education.

The framework classifies learners into three competency tiers — Novice, Developing, and Proficient — using a stacked ensemble of Random Forest, XGBoost, and SVM-RBF base models, with SHAP feature attributions injected as meta-features into a Logistic Regression meta-learner. LIME-based counterfactual feedback is delivered to learners as actionable guidance.

---

## Research Context

| Field | Details |
|---|---|
| Author | Thomas Dzelu |
| Supervisor | Dr. Linda Aurelia Ofori |
| Institution | Department of Computer Science, KNUST, Kumasi, Ghana |
| Degree | Master of Philosophy (MPhil) in Computer Science |
| SDG Alignment | SDG 4 — Quality Education (Targets 4.4 and 4.7) |

---

## The CASTLE Framework — 7-Layer Architecture

    Layer 1 — Data Ingestion and Learner Profiling
    Layer 2 — Privacy and Ethics (k-anonymity, GDPR, RBAC)
    Layer 3 — Feature Engineering (SMOTE, MI, RFE)
    Layer 4 — Explanation-Guided Stacked Ensemble [CORE]
               RF (TreeExplainer SHAP)
               XGBoost (TreeExplainer SHAP)
               SVM-RBF (KernelExplainer SHAP)
               Logistic Regression Meta-Learner
    Layer 5 — Competency Gap Analysis Engine (SHAP-driven)
    Layer 6 — Adaptive Content Recommendation (LIME feedback)
    Layer 7 — Monitoring and Visualisation Dashboard

---

## Key Results

### Machine Learning Performance (Test Set, n=1,659)

| Model | Macro-F1 | Accuracy | AUC-ROC |
|---|---|---|---|
| CASTLE (proposed) | 0.9812 | 0.9795 | 0.9983 |
| SVM-RBF | 0.9967 | 0.9958 | 1.0000 |
| VanillaStacking | 0.9812 | 0.9795 | 0.9971 |
| XGBoost | 0.9698 | 0.9693 | 0.9985 |
| LogisticReg | 0.9624 | 0.9620 | 0.9982 |
| KNN | 0.9429 | 0.9415 | 0.9904 |
| RF | 0.9291 | 0.9307 | 0.9916 |
| DecisionTree | 0.8913 | 0.8927 | 0.9374 |
| NaiveBayes | 0.8879 | 0.8849 | 0.9731 |

### Per-Class Results (CASTLE)

| Tier | Precision | Recall | F1 |
|---|---|---|---|
| Novice | 1.0000 | 0.9740 | 0.9868 |
| Developing | 0.9736 | 0.9791 | 0.9764 |
| Proficient | 0.9790 | 0.9817 | 0.9803 |

### Pilot Study (n=186)

| Metric | Experimental | Control |
|---|---|---|
| Pre-test mean | 50.72 | 52.63 |
| Post-test mean | 68.13 | 61.45 |
| Gain mean | 17.42 | 8.82 |
| Cohen d (within) | 1.5609 | 1.1432 |

Between-group: t=2.7279, p=0.0070, d=0.4022 (significant, p < 0.05)

Cronbach alpha = 0.9738 (20-item competency instrument)

### Educator Trust (TAM Survey, n=93)

| Subscale | Mean (1-5) |
|---|---|
| Perceived Usefulness | 4.15 |
| Perceived Ease of Use | 3.92 |
| Trust in Explanations | 4.15 |
| Intention to Use | 4.07 |

---

## Global SHAP Feature Importance

Top predictors ranked by mean absolute SHAP value:

1. password_score
2. social_eng_score
3. secure_browsing_score
4. phishing_score
5. data_privacy_score
6. correct_identification
7. click_rate
8. self_efficacy
9. recognition_latency_s
10. behaviour_intention
11. threat_perception

---

## Dataset

| Source | Description |
|---|---|
| OULAD (Kaggle) | Open University Learning Analytics — real LMS interaction logs |
| Phishing Website Detector (UCI/Kaggle) | Real cybersecurity feature dataset |

Combined: 11,054 learners, 25 raw features, 11 selected, 3 competency tiers

---

## Repository Structure

    Castle_final/
    ├── data/
    │   ├── raw/                   # Downloaded datasets (not committed)
    │   └── processed/             # Train/val/test splits
    ├── models/                    # Saved model checkpoints (.pkl)
    ├── figures/                   # All thesis figures (PNG, 300 DPI)
    ├── results/                   # All results JSON files
    │   ├── dataset_stats.json
    │   ├── selected_features.json
    │   ├── base_model_params.json
    │   ├── castle_results.json
    │   ├── baseline_ablation_results.json
    │   ├── pilot_results.json
    │   └── all_paper_numbers.json
    └── notebooks/
        ├── 01_Data_Preprocessing.ipynb
        ├── 02_Feature_Engineering.ipynb
        ├── 03_Base_Model_Training.ipynb
        ├── 04_CASTLE_Stacking.ipynb
        ├── 05_Baselines_Ablation.ipynb
        ├── 06_Pilot_Study.ipynb
        └── 07_Final_Results_Summary.ipynb

---

## Technology Stack

| Component | Technology |
|---|---|
| Base Learners | scikit-learn (RF, SVM), XGBoost |
| SHAP Extraction | shap (TreeExplainer, KernelExplainer) |
| Hyperparameter Tuning | Optuna (Bayesian, 50 trials per model) |
| Class Imbalance | imbalanced-learn (SMOTE, k=5) |
| Meta-Learner | scikit-learn LogisticRegression |
| Compute | Google Colab Pro (NVIDIA L4 GPU) |
| Visualisation | Matplotlib, Seaborn |

---

## Reproducibility

All experiments use random_state=42 and np.random.seed(42) throughout.

    git clone https://github.com/NehlTech/Castle_final.git
    cd Castle_final
    pip install scikit-learn xgboost shap optuna imbalanced-learn pandas numpy matplotlib seaborn joblib scipy
    # Run notebooks 01 to 07 in order in Google Colab Pro

---

## Research Contributions

1. Architectural: A novel seven-layer framework embedding SHAP-enriched stacked ensemble learning within a closed-loop adaptive tutoring architecture.
2. Methodological: A domain-specific feature engineering pipeline for digital safety literacy incorporating phishing simulation performance, threat recognition latency, and secure-behaviour self-efficacy.
3. Empirical: Quasi-experimental evidence that explanation-driven adaptive instruction significantly improves learner digital safety competency (t=2.73, p=0.007, d=0.40).
4. Transferable: A reusable architecture applicable to financial literacy, health literacy, and data privacy awareness education.

---

## Citation

    @mastersthesis{dzelu2026castle,
      author     = {Dzelu, Thomas},
      title      = {Design of an Explainable Stacked Ensemble Learning Framework
                    for Adaptive Digital Safety Competency Prediction in
                    Smart Learning Environments},
      school     = {Kwame Nkrumah University of Science and Technology},
      year       = {2026},
      type       = {MPhil Thesis},
      address    = {Kumasi, Ghana},
      supervisor = {Dr. Linda Aurelia Ofori}
    }

---

## License

MIT License. See LICENSE for details.

---

## Acknowledgements

Supervised by Dr. Linda Aurelia Ofori, Department of Computer Science, KNUST.
Built upon the XStacking paradigm (Garouani et al., 2025, Information Fusion).
Data: Open University Learning Analytics Dataset and UCI Phishing Website Detector Dataset.
