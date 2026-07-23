# Comprehensive Project Report: Model Fairness, Bias, and Explainability Analysis

## 1. Project Title & Overview
This project focuses on the ethics and transparency of Machine Learning. Using a Random Forest model trained on telecom customer churn data, we conduct a deep dive into model explainability and bias detection to ensure fairness across sensitive demographics.

## 2. Dataset Description
- **Source:** `customer_churn_data.csv` (Telecom dataset).
- **Target Variable:** Churn (Yes/No).
- **Sensitive Attributes Analyzed:** `gender` (Male/Female) and `SeniorCitizen` (0/1).

## 3. Setup & Installation
### Prerequisites
- Python 3.8+
### Installation
Navigate to this directory and install the required dependencies:
```bash
pip install -r requirements.txt
```

## 4. Explainability Analysis
To demystify the "black box" nature of the Random Forest:
- **Global Explainability (SHAP):** We used SHAP (SHapley Additive exPlanations) to understand feature importance across the entire dataset. For instance, determining how much high 'tenure' overall reduces churn risk.
- **Local Explainability (LIME):** We used LIME (Local Interpretable Model-agnostic Explanations) to explain individual predictions, showing exactly which features drove the decision for a specific customer.

## 5. Bias Detection & Mitigation
- **Detection:** We evaluated False Positive Rates (FPR) and True Positive Rates (TPR) across demographic splits. The baseline model exhibited a bias, showing higher False Positive Rates for Males and Senior Citizens.
- **Mitigation (SMOTE):** To counteract this, we applied **SMOTE** (Synthetic Minority Over-sampling Technique). By resynthesizing the training data to balance class representations, we successfully reduced the FPR disparity between demographics without sacrificing overall accuracy.

## 6. Usage / How to Run
You can run the full fairness analysis and see the explainability plots by executing the script:
```bash
# To run the script version:
python task3.py
```
*(Refer to the screenshots in this directory for visual outputs of the SHAP and LIME visualizations).*
