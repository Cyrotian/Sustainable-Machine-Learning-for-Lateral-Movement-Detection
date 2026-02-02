# Sustainable Machine Learning for Lateral Movement Detection

A comparative analysis of machine learning models for detecting lateral movement in cybersecurity networks, evaluated through the lens of environmental sustainability.

## Overview

This research evaluates three machine learning models—Random Forest, Isolation Forest, and LSTM—for lateral movement detection using both traditional performance metrics and environmental impact measures. The project introduces a composite "Sustainability Score" that balances detection accuracy with carbon footprint considerations.

## Research Objectives

- Compare ML model performance for lateral movement detection using traditional metrics (F1-score, inference time)
- Measure environmental impact (energy consumption, CO₂ emissions) using CodeCarbon
- Develop a sustainability scoring framework to identify optimal models that balance performance and environmental cost
- Provide evidence-based recommendations for sustainable ML deployment in cybersecurity

## Datasets

- **LMD-2022**: Lateral Movement Detection dataset (2022)
- **LMD-2023**: Updated Lateral Movement Detection dataset (2023)

## Models Evaluated

1. **Random Forest**: Ensemble learning method for classification
2. **Isolation Forest**: Anomaly detection algorithm
3. **LSTM** (Long Short-Term Memory): Recurrent neural network for sequence analysis

## Evaluation Metrics

### Traditional Performance Metrics
- F1-Score
- Inference Time
- Precision & Recall

### Sustainability Metrics
- Energy Consumption (kWh)
- CO₂ Emissions (kg CO₂eq)
- Carbon Intensity

### Composite Score
The Sustainability Score uses Simple Additive Weighting (SAW) methodology:
```
Sustainability Score = w₁(F1-Score) + w₂(1 - Normalized Energy) + w₃(1 - Normalized CO₂)
```

## Installation
```bash
# Clone the repository
git clone https://github.com/cyrotian/sustainable-ml-lateral-movement.git
cd sustainable-ml-lateral-movement

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements
```
pandas
numpy
scikit-learn
tensorflow
codecarbon
matplotlib
seaborn
```

## Usage
```bash
# Train and evaluate models
python train_models.py --dataset lmd2022

# Compare model sustainability
python evaluate_sustainability.py

# Generate results report
python generate_report.py
```

## Project Structure
```
├── data/
│   ├── lmd2022/
│   └── lmd2023/
├── models/
│   ├── random_forest.py
│   ├── isolation_forest.py
│   └── lstm.py
├── evaluation/
│   ├── performance_metrics.py
│   └── sustainability_metrics.py
├── results/
│   ├── emissions/
│   └── reports/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── train_models.py
├── evaluate_sustainability.py
├── requirements.txt
└── README.md
```

## Methodology

The research follows a structured approach:

1. **Data Preprocessing**: Feature extraction and normalization from LMD datasets
2. **Model Training**: Training each model with consistent hyperparameters
3. **Performance Evaluation**: Traditional ML metrics assessment
4. **Carbon Tracking**: CodeCarbon integration for emissions measurement
5. **Sustainability Scoring**: SAW methodology for composite scoring
6. **Comparative Analysis**: Multi-criteria evaluation across models

## Key Findings

*Results will be updated as experiments are completed*

## References

- Usman et al. - Systems Theory Model for sustainability assessment
- Green Algorithms - Carbon footprint measurement framework
- MITRE ATT&CK - Lateral Movement tactics and techniques

## Author

**Micheal Olanegan** - Computer Science Undergraduate, Brunel University

Dissertation Project - 2024/2025

## License

This project is part of academic research at Brunel University.

## Acknowledgments

- Brunel University Computer Science Department
- Dissertation Supervisor
- LMD Dataset Contributors
