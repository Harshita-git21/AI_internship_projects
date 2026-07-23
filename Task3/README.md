# Model Fairness, Bias, and Explainability Analysis
## Environment Setup
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   To get rid of errors, ensure all the necessary modules are installed. Run the following command:
   ```
   pip install -r requirements.txt
   ```
   This will install the following required modules:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - shap
   - lime
   - imbalanced-learn
   - jupyter

## Running the Analysis
1. Open `task3.ipynb` in Jupyter/VSCode.
2. Run all cells sequentially.
   - Cells 1-2: Data loading, preprocessing, model training (RandomForestClassifier).
   - Cell 3: Feature importance plot.
   - Cell 4: Install additional libs if needed.
   - Cells 5-7: SHAP global summary, bar plot, force plot (local explanation).
   - Cell 8: LIME local explanation.
   - Cells 9-11: Basic bias checks (gender accuracy, prediction rate, FPR).
   - Cells 12-15: SMOTE mitigation, post-mitigation performance.
   - Final cell: Complete analysis with bias tables/plots before/after mitigation.

## Key Outputs
- SHAP summary plots: Global feature impacts.
- LIME: Local prediction explanations.
- Bias metrics: FPR/TPR/Accuracy/Prediction Rate by gender_Male (0=Female,1=Male), SeniorCitizen (0/1).
- Mitigation: SMOTE reduces disparity.

## Screenshots
- screenshot 1.png, Screenshot 2.png: Existing model outputs.
- Run notebook for new plots (SHAP/LIME/bias comparisons).

Dataset: `customer_churn_data.csv` (Telco churn, sensitive attrs: gender, SeniorCitizen).

No code modifications needed - analysis fully implemented in notebook.

