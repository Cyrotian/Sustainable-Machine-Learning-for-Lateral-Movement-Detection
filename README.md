# Sustainable Machine Learning for Lateral Movement Detection

A comparative study of Random Forest, LightGBM and LSTM for detecting lateral movement in enterprise networks, examining detection performance, computational efficiency and carbon emissions.

## Overview

Lateral movement is a technique attackers use to move between systems after gaining access to a network. Machine learning can help detect this activity, but more complex models also demand more computational resources.

This project investigates whether that additional complexity leads to meaningful improvements in detection performance.

Three models — **Random Forest, LightGBM and LSTM** — were trained on LMD-2022 and evaluated on LMD-2023 to assess their ability to generalise across datasets collected in different years.

Alongside conventional classification metrics, I measured inference time, energy consumption and CO₂ emissions using CodeCarbon. I also tested different model configurations to examine how increasing complexity affects both detection performance and environmental cost.

The main finding was that **Random Forest and LightGBM achieved their best F1 scores at the lowest tested complexity**, while increasing LSTM complexity raised emissions without proportionate improvements in detection.

A composite Sustainability Score was also developed to compare models under different operational priorities.

## Research Objectives

- Compare Random Forest, LightGBM and LSTM using F1-score, precision, recall, false positive rate and inference time.
- Measure inference-phase energy consumption and CO₂ emissions using CodeCarbon.
- Examine how model complexity affects detection performance, execution time and emissions.
- Develop a Sustainability Score using Simple Additive Weighting (SAW) to support model comparisons.
- Assess the trade-offs involved in deploying machine learning for lateral movement detection in Security Operations Centres (SOCs).

## Datasets

Two lateral movement detection datasets were used:

| Dataset | Purpose |
|---|---|
| **LMD-2022** | Model training and feature selection |
| **LMD-2023** | Independent cross-year testing |

Training and testing on separate years provides a more demanding evaluation than a random split of a single dataset. It also helps reduce the risk of testbed-specific information influencing the reported performance.

## Models Evaluated

| Model | Approach | Description |
|---|---|---|
| **Random Forest** | Bagging ensemble | Combines multiple decision trees for classification |
| **LightGBM** | Gradient boosting | Builds decision trees sequentially to correct previous errors |
| **LSTM** | Recurrent neural network | Neural network architecture designed to capture sequential patterns |

The comparison examines whether the additional complexity of neural networks is justified by improvements in detection performance.

## Feature Engineering and Selection

The datasets were cleaned and transformed before training. Preprocessing included grouping rare categories, encoding categorical variables and maintaining consistent feature representations across training and testing.

Feature selection was based on Random Forest importance.

Mutual Information was also explored, but it prioritised `EventID`, which led to overfitting. Random Forest importance was therefore preferred.

The final five features were:

1. `Image_Category`
2. `GrantedAccess`
3. `RuleName`
4. `Signed`
5. `SourceImage`

High-cardinality categorical variables were handled using rare-category grouping, with consistent label encoding across both datasets.

## Evaluation Metrics

### Detection Performance

The models were evaluated using:

- **F1-score:** Balances precision and recall.
- **Precision:** Measures how many predicted attacks were correctly classified.
- **Recall:** Measures how many actual attacks were detected.
- **False Positive Rate (FPR):** Measures how often normal activity was incorrectly classified as malicious.
- **Inference Time:** Measures the time required to generate predictions.

### Environmental Impact

Environmental measurements focused on the inference phase rather than model training.

Using CodeCarbon, I recorded:

- Energy consumption (kWh)
- CO₂ emissions (kg CO₂eq)
- Carbon intensity

Each model underwent **10 inference iterations** to collect emissions measurements.

### Sustainability Score

A composite Sustainability Score (SS) was developed using **Simple Additive Weighting (SAW)**, informed by Usman et al.'s Systems Theory Model.

The score combines detection performance, false positives, inference time and carbon emissions:

`SS = α × F1 + β × (1 − FPR_norm) + γ × (1 − Time_norm) + δ × (1 − CO₂_norm)`

Where:

- `F1` represents detection performance.
- `FPR_norm` represents normalised false positive rate.
- `Time_norm` represents normalised inference time.
- `CO₂_norm` represents normalised emissions.
- `α, β, γ, δ` determine the relative importance of each component.

The baseline assigns equal weights:

`α = β = γ = δ = 0.25`

A sensitivity analysis then examines how model rankings change under different operational priorities.

| Scenario | Main Priority |
|---|---|
| **Balanced** | Equal weighting across all four metrics |
| **Green First** | Minimise carbon emissions |
| **Detection First** | Maximise detection performance |
| **SOC Operational** | Prioritise false positives and inference time |

The score is intended as a supporting comparison tool, not a replacement for individual performance and emissions measurements.

## Methodology

The experiment followed eight stages.

**1. Data preprocessing**

Cleaned the datasets, engineered features, grouped rare categories and encoded categorical variables.

**2. Feature selection**

Compared feature selection approaches and retained the five highest-priority features identified using Random Forest importance.

**3. Model training**

Trained Random Forest, LightGBM and LSTM on LMD-2022 using a consistent preprocessing approach.

**4. Cross-year evaluation**

Tested the trained models on LMD-2023 to examine generalisation beyond the training dataset.

**5. Carbon measurement**

Used CodeCarbon to measure energy consumption and CO₂ emissions across 10 inference iterations per model.

**6. Complexity–emissions analysis**

Varied model configurations to examine whether additional complexity improved detection enough to justify the associated increase in computational and environmental cost.

**7. Ablation study**

Investigated potential Infra/Splunk confounding effects and their influence on the evaluation.

**8. Sustainability scoring**

Combined normalised performance and environmental metrics using SAW, then compared rankings under different weighting scenarios.

## Key Findings

- **Model complexity did not consistently improve detection.** Random Forest and LightGBM achieved their highest F1 scores at the minimum tested configuration.
- **Increasing complexity raised emissions.** More computationally expensive configurations did not always produce meaningful detection improvements.
- **LSTM showed diminishing returns.** Additional model complexity increased environmental cost without proportional performance gains.
- **The preferred model depends on operational priorities.** Detection accuracy, false positives, inference speed and emissions can lead to different model rankings.

These findings suggest that model selection for cybersecurity should consider computational cost alongside detection performance, rather than assuming that more complex architectures will always deliver better results.

## Tools and Technologies

- **Python:** Data processing, feature engineering and experiments
- **pandas and NumPy:** Data cleaning and manipulation
- **scikit-learn:** Random Forest, feature selection and classification metrics
- **LightGBM:** Gradient-boosted decision trees
- **TensorFlow/Keras:** LSTM implementation
- **CodeCarbon:** Energy and carbon-emissions measurement

## Research Context

This project was completed as a final-year BSc Computer Science dissertation at Brunel University of London (2025–2026).

The research focuses on the trade-off between detecting potentially malicious network activity and the computational resources required to do so. The aim is to support more informed decisions about machine learning deployment in security monitoring environments.
