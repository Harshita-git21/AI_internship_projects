# Comprehensive Project Report: Customer Churn Classification

## 1. Project Title & Overview
This project focuses on building a predictive machine learning model to identify customers at high risk of churning from a telecom service. By accurately identifying these individuals, the business can take proactive retention measures and minimize revenue loss.

## 2. Dataset Description
- **Source:** `customer_churn_data.csv`
- **Features:** Includes customer demographics, account information, and subscribed services (e.g., tenure, MonthlyCharges, Contract type).
- **Target Variable:** `Churn` (Binary: Yes/No indicating if the customer left).

## 3. Setup & Installation
### Prerequisites
- Python 3.8+
### Installation
Navigate to this directory and install the required dependencies:
```bash
pip install -r requirements.txt
```

## 4. Methodology & Pipeline
- **Data Preprocessing:** 
  - Dropped non-predictive columns (`customerID`).
  - Handled missing values (imputed median for `TotalCharges`).
  - Scaled numerical features using `StandardScaler`.
  - Encoded categorical variables using `OneHotEncoder`.
- **Model Architecture:** We trained and compared three different models to find the best fit:
  1. Logistic Regression (Baseline)
  2. Random Forest
  3. Gradient Boosting Classifier

## 5. Results & Evaluation
The models were evaluated using 5-fold stratified cross-validation and a hold-out test set. 
- **Metrics Evaluated:** Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- **Winning Model:** The **Random Forest** model was selected as the best-performing model based on achieving the highest ROC-AUC score.
*(Refer to `models_performance_comparison.png` and `Screenshot.png` in this folder for visual performance comparisons).*

## 6. Usage / How to Run
You can run the full pipeline either through the Jupyter Notebook or the provided Python script:
```bash
# To run the script version:
python task1.py
```

## 7. Appendix: Screenshots & Visualizations

*(Attach `models_performance_comparison.png` in the space below)*

<div style="width: 100%; height: 350px; border: 2px dashed #ccc; margin-bottom: 20px; text-align: center; line-height: 350px; color: #999;">
  [ Space for models_performance_comparison.png ]
</div>

*(Attach `Screenshot.png` in the space below)*

<div style="width: 100%; height: 350px; border: 2px dashed #ccc; margin-bottom: 20px; text-align: center; line-height: 350px; color: #999;">
  [ Space for Screenshot.png ]
</div>
