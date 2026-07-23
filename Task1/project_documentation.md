# Project Documentation: Customer Churn Classification

## Project Overview

This document provides a detailed overview of the Customer Churn Classification project. The goal of this project is to analyze customer data and build a machine learning model to predict customer churn. This helps in identifying customers who are likely to leave the service, enabling proactive measures to retain them.

## Code and Notebooks

*   **Jupyter Notebook:** The complete analysis, including data loading, preprocessing, model training, and evaluation, is contained in the following notebook:
    *   [Customer_Churn_Classification.ipynb](Customer_Churn_Classification.ipynb)

## Screenshots

This project generates several visualizations to understand the data and model performance. Please add the following screenshots to this document:

*   `Screenshot.png`: A general screenshot of the project.
*   `models_performance_comparison.png`: A comparison of the performance of the different models.

## Explanation

### 1. Data Loading and Exploration

*   The project starts by loading the `customer_churn_data.csv` dataset.
*   Initial data exploration is performed to understand the dataset's structure, including the shape, column names, and data types.
*   A statistical summary of numerical features and the distribution of the target variable ('Churn') are examined.

### 2. Data Preprocessing

*   The `customerID` column is removed as it's not a useful feature for prediction.
*   The target variable `Churn` is converted into a binary format (1 for 'Yes', 0 for 'No').
*   `TotalCharges` is converted to a numeric type, and any missing values are filled with the median.
*   Features are separated into categorical and numerical lists for tailored preprocessing.

### 3. Feature Engineering and Train/Test Split

*   The data is split into features (X) and target (y).
*   The dataset is divided into training (80%) and testing (20%) sets using a stratified split to maintain the same proportion of churned and non-churned customers in both sets.
*   A preprocessing pipeline is created using `ColumnTransformer`.
    *   **Categorical Features:** `OneHotEncoder` is used to convert categorical variables into a numerical format.
    *   **Numerical Features:** `StandardScaler` is used to scale numerical features to have a mean of 0 and a standard deviation of 1.

### 4. Model Training and Evaluation

*   Three different classification models are trained and compared:
    1.  **Logistic Regression**
    2.  **Random Forest**
    3.  **Gradient Boosting**
*   **Cross-Validation:** 5-fold stratified cross-validation is used to evaluate the models on the training data. The models are evaluated on a variety of metrics: Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
*   **Test Set Evaluation:** The trained models are then used to make predictions on the unseen test set. The performance on the test set is used as the final evaluation of the models.

### 5. Results

*   The performance of all three models is compared based on the evaluation metrics from both cross-validation and the test set.
*   Confusion matrices and ROC curves are plotted to visualize the performance of the models on the test set.
*   Based on the ROC-AUC score on the test set, the **Random Forest** model was selected as the best-performing model for this task.
