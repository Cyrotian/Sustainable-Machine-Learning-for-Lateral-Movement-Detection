# Sustainable Machine Learning for Lateral Movement Detection

A comparative analysis of machine learning models for detecting lateral movement in enterprise networks, evaluated through the dual lens of detection performance and environmental sustainability.

## Overview

This research evaluates three machine learning models: **Random Forest**, **LightGBM**, and **LSTM** for lateral movement detection using both traditional performance metrics and environmental impact measures. Models are trained on LMD-2022 and tested on LMD-2023 in a cross-year generalisation design

A key empirical finding is that tree-based models (Random Forest, LightGBM) achieve peak F1 at minimum configuration, while LSTM scales emissions without meaningful performance gains. The **Sustainability Score** (SS) is introduced as a supporting analytical tool to surface these trade-offs, rather than as the primary contribution.

## Research Objectives

- Compare ML model performance for lateral movement detection using traditional metrics (F1-score, FPR, inference time)
- Measure environmental impact (energy consumption, CO₂ emissions) during inference using CodeCarbon
- Conduct a complexity-emissions sweep to analyse how model configuration affects the performance/sustainability trade-off
- Develop a composite Sustainability Score (SAW-based) to support multi-criteria model selection
- Provide evidence-based recommendations for sustainable ML deployment in cybersecurity SOC contexts

## Datasets

| Dataset | Description |
|---|---|
| **LMD-2022** | Lateral Movement Detection dataset - used for training |
| **LMD-2023** | Updated Lateral Movement Detection dataset - used for testing |

The cross-year train/test split tests genuine temporal generalisation, guarding against the testbed leakage artefacts common in prior work (e.g., Smiliotopoulos et al., 2023).

## Models Evaluated

| Model | Type | Role |
|---|---|---|
| **Random Forest** | Ensemble (bagging) | Primary classifier |
| **LightGBM** | Gradient boosted trees | Primary classifier |
| **LSTM** | Recurrent neural network | Sequence-aware classifier |

> Isolation Forest was evaluated but abandoned due to poor performance on feature-sparse benign-only training data.

## Feature Set

Five features were selected via Random Forest importance (preferred over Mutual Information, which caused overfitting by selecting EventID):

1. `Image_Category`
2. `GrantedAccess`
3. `RuleName`
4. `Signed`
5. `SourceImage`

High-cardinality categorical features use rare-category grouping; LabelEncoder consistency is maintained across train/test splits.

## Evaluation Metrics

### Traditional Performance Metrics
- F1-Score
- False Positive Rate (FPR)
- Precision & Recall
- Inference Time

### Sustainability Metrics
- Energy Consumption (kWh) - inference phase only
- CO₂ Emissions (kg CO₂eq) - 10 inference iterations per model
- Carbon Intensity

### Composite Sustainability Score

The SS uses Simple Additive Weighting (SAW) with four components, grounded in Usman et al.'s Systems Theory Model:

```
SS = α·F1 + β·(1 − FPR_norm) + γ·(1 − Time_norm) + δ·(1 − CO₂_norm)
```

Baseline weights: `α = β = γ = δ = 0.25`

Sensitivity analysis across scenarios:

| Scenario | Priority |
|---|---|
| **Green First** | Minimise CO₂ |
| **Detection First** | Maximise F1 |
| **SOC Operational** | Balance FPR + inference time |
| **Balanced** | Equal weighting (baseline) |

## Methodology

1. **Data Preprocessing** - Feature extraction, rare-category grouping, label encoding
2. **Feature Selection** - Random Forest importance (top 5 features)
3. **Model Training** - LMD-2022; consistent preprocessing pipeline
4. **Cross-Year Testing** - Evaluation on LMD-2023 for generalisation
5. **Carbon Tracking** - CodeCarbon for inference-phase emissions (10 iterations)
6. **Complexity-Emissions Sweep** - Configuration variation to map performance/sustainability frontier
7. **Ablation Study** - Infra/Splunk confound analysis
8. **Sustainability Scoring** - SAW with sensitivity analysis across weight scenarios

## Key Findings

- Tree models (RF, LightGBM) reach peak F1 at minimum configuration - additional complexity yields no gains
- LSTM emissions scale with configuration without corresponding F1 improvement
- The complexity-emissions sweep reframes the SS as a decision-support tool for identifying the efficient frontier
- Tree models show greater sensitivity to the Infra/Splunk testbed artefact than LSTM


## References

- Usman et al. - Systems Theory Model for sustainability assessment
- Lannelongue et al. - Green Algorithms: carbon footprint measurement framework
- Smiliotopoulos et al. (2023) - Lateral movement detection benchmark (within-dataset CV)
- MITRE ATT&CK - Lateral Movement tactics and techniques (TA0008)

## Author

**Michael Olanegan** - BSc Computer Science, Brunel University London  

## License

This project is part of academic research at Brunel University London.
