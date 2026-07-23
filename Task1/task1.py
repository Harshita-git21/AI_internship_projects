import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)


import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')

# Display basic information

print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:")
df.head()

info_df = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
    'Unique Values': df.nunique()
})
info_df

df.describe()

churn_dist = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 10

print(f"\nChurn Distribution:")
print(f"  No Churn:  {churn_dist['No']} ({churn_pct['No']:.2f}%)")
print(f"  Churn:     {churn_dist['Yes']} ({churn_pct['Yes']:.2f}%)")

# Visualize target distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
sns.countplot(data=df, x='Churn', palette='viridis', ax=ax[0])
ax[0].set_title('Churn Distribution (Count)', fontsize=12)
ax[0].set_xlabel('Churn')
ax[0].set_ylabel('Count')

# Pie chart
colors = ['#2ecc71', '#e74c3c']
ax[1].pie(churn_dist, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
          colors=colors, explode=(0, 0.1), shadow=True)
ax[1].set_title('Churn Distribution (Percentage)', fontsize=12)

plt.tight_layout()
plt.show()

# Step 1: Remove customerID (not useful for prediction)
df_processed = df.drop(columns=['customerID'])
print("\n1. Removed 'customerID' column (not useful for prediction)")

# Step 2: Convert target variable to binary
df_processed['Churn'] = df_processed['Churn'].replace({'Yes': 1, 'No': 0})
print("2. Converted 'Churn' to binary (Yes=1, No=0)")

# Step 3: Handle TotalCharges - convert to numeric (some values might be empty)
df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with median
median_charges = df_processed['TotalCharges'].median()
df_processed['TotalCharges'].fillna(median_charges, inplace=True)
print(f"3. Converted 'TotalCharges' to numeric, filled {df['TotalCharges'].isnull().sum()} missing values with median")

# Step 4: Identify categorical and numerical features
cat_features = df_processed.select_dtypes(include=['object']).columns.tolist()
num_features = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Safe removal of target from features
if 'Churn' in num_features:
    num_features.remove('Churn')

print(f"\n4. Identified features:")
print(f"   Categorical ({len(cat_features)}): {cat_features}")
print(f"   Numerical ({len(num_features)}): {num_features}")

# Display processed data info
print(f"\n5. Processed dataset shape: {df_processed.shape}")
print("\n✓ Data preprocessing complete!")

# Visualize categorical feature distributions
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for idx, col in enumerate(cat_features):
    if idx < 16:
        churn_rate = df_processed.groupby(col)['Churn'].mean().sort_values(ascending=False)
        sns.barplot(x=churn_rate.index, y=churn_rate.values, palette='coolwarm', ax=axes[idx])
        axes[idx].set_title(f'Churn Rate by {col}', fontsize=10)
        axes[idx].set_ylabel('Churn Rate')
        axes[idx].tick_params(axis='x', rotation=45)

# Remove empty subplots
for idx in range(len(cat_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

# Numerical features distribution by churn
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, col in enumerate(num_features):
    if idx < 6:
        sns.histplot(data=df_processed, x=col, hue='Churn', kde=True, ax=axes[idx], palette='coolwarm')
        axes[idx].set_title(f'{col} Distribution by Churn', fontsize=10)

plt.tight_layout()
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(10, 8))
corr_matrix = df_processed[num_features + ['Churn']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
            linewidths=0.5, square=True)
plt.title('Correlation Heatmap - Numerical Features vs Churn', fontsize=12)
plt.tight_layout()
plt.show()

# Show correlation with target
print("\nCorrelation with Churn:")
print(corr_matrix['Churn'].sort_values(ascending=False))

# Prepare features and target
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

# Train/Test Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution
)

print("=" * 60)
print("TRAIN/TEST SPLIT")
print("=" * 60)
print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nTraining set churn distribution:")
print(f"  No Churn:  {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.2f}%)")
print(f"  Churn:     {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.2f}%)")
print(f"\nTest set churn distribution:")
print(f"  No Churn:  {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.2f}%)")
print(f"  Churn:     {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.2f}%)")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ('num', StandardScaler(), num_features)
    ]
)

print("=" * 60)
print("PREPROCESSING PIPELINE")
print("=" * 60)
print("\n✓ Categorical features: OneHotEncoder")
print("✓ Numerical features: StandardScaler")

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Cross-validation results storage
cv_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Create pipeline with preprocessor and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    # Store mean scores
    cv_results[name] = {
        'Accuracy': scores['test_accuracy'].mean(),
        'Precision': scores['test_precision'].mean(),
        'Recall': scores['test_recall'].mean(),
        'F1 Score': scores['test_f1'].mean(),
        'ROC-AUC': scores['test_roc_auc'].mean()
    }

    print(f"  ✓ {name} completed")

# Display cross-validation results
cv_results_df = pd.DataFrame(cv_results).T
print("\n" + "=" * 60)
print("CROSS-VALIDATION METRICS SUMMARY")
print("=" * 60)
print(cv_results_df.round(4))

# Visualize cross-validation results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
colors = ['#3498db', '#2ecc71', '#e74c3c']

# Bar chart for all metrics
x = np.arange(len(models))
width = 0.15

for idx, metric in enumerate(metrics):
    row = idx // 3
    col = idx % 3

    values = [cv_results[model][metric] for model in cv_results.keys()]
    bars = axes[row, col].bar(x, values, color=colors, alpha=0.8)
    axes[row, col].set_xlabel('Model')
    axes[row, col].set_ylabel(metric)
    axes[row, col].set_title(f'{metric} Comparison', fontsize=11)
    axes[row, col].set_xticks(x)
    axes[row, col].set_xticklabels(list(cv_results.keys()), rotation=15, ha='right')
    axes[row, col].set_ylim(0, 1)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{val:.3f}', ha='center', fontsize=8)

# Hide empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Train models on full training set and evaluate on test set
test_results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} on full training set...")

    # Create and train pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    test_results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }

    # Store trained model
    trained_models[name] = pipeline

    print(f"  ✓ {name} trained and evaluated")

# Display test results
test_results_df = pd.DataFrame(test_results).T
print("\n" + "=" * 60)
print("TEST SET METRICS SUMMARY")
print("=" * 60)
print(test_results_df.round(4))

comparison_df = pd.DataFrame({
    'Model': list(test_results.keys()),
    'CV Accuracy': [cv_results[m]['Accuracy'] for m in test_results.keys()],
    'Test Accuracy': [test_results[m]['Accuracy'] for m in test_results.keys()],
    'CV ROC-AUC': [cv_results[m]['ROC-AUC'] for m in test_results.keys()],
    'Test ROC-AUC': [test_results[m]['ROC-AUC'] for m in test_results.keys()]
})
print(comparison_df.round(4))

# Confusion Matrix for all models
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, pipeline) in enumerate(trained_models.items()):
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=11)

plt.tight_layout()
plt.show()

# ROC Curves for all models
plt.figure(figsize=(10, 8))

colors = ['#3498db', '#2ecc71', '#e74c3c']

for idx, (name, pipeline) in enumerate(trained_models.items()):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.plot(fpr, tpr, color=colors[idx], lw=2,
             label=f'{name} (AUC = {auc_score:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Select best model based on ROC-AUC
best_model_name = max(test_results.keys(), key=lambda x: test_results[x]['ROC-AUC'])
best_pipeline = trained_models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print("\nPerformance Metrics:")
for metric, value in test_results[best_model_name].items():
    print(f"  {metric}: {value:.4f}")

# Feature Importance (for Random Forest and Gradient Boosting)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Get feature names after preprocessing
    feature_names = (list(trained_models[best_model_name].named_steps['preprocessor']
                          .named_transformers_['cat'].get_feature_names_out(cat_features)) +
                     num_features)

    # Get feature importances
    importances = best_pipeline.named_steps['classifier'].feature_importances_

    # Sort and display top 15 features
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top 15 Feature Importance - {best_model_name}', fontsize=12)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
#     plt.show()

    print("\nTop 15 Most Important Features:")
    print(feature_importance_df.to_string(index=False))

# Create test cases for prediction
test_cases = pd.DataFrame({
    'gender': ['Male', 'Female', 'Male', 'Female'],
    'SeniorCitizen': [0, 1, 0, 1],
    'Partner': ['Yes', 'No', 'Yes', 'No'],
    'Dependents': ['No', 'No', 'Yes', 'Yes'],
    'tenure': [72, 6, 24, 48],
    'PhoneService': ['Yes', 'Yes', 'No', 'Yes'],
    'MultipleLines': ['No', 'Yes', 'No phone service', 'No'],
    'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service', 'Yes'],
    'OnlineBackup': ['Yes', 'No', 'No internet service', 'Yes'],
    'DeviceProtection': ['Yes', 'No', 'No internet service', 'No'],
    'TechSupport': ['Yes', 'No', 'No internet service', 'Yes'],
    'StreamingTV': ['No', 'Yes', 'No internet service', 'No'],
    'StreamingMovies': ['No', 'Yes', 'No internet service', 'No'],
    'Contract': ['Two year', 'Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes', 'No', 'Yes'],
    'PaymentMethod': ['Bank transfer', 'Electronic check', 'Credit card', 'Mailed check'],
    'MonthlyCharges': [35.00, 89.99, 55.50, 75.00],
    'TotalCharges': [2520.00, 540.00, 1332.00, 3600.00]
})

print("=" * 60)
print("TEST CASES FOR PREDICTION")
print("=" * 60)
print("\nTest Case Data:")
print(test_cases.to_string())

# Make predictions using the best model
predictions = best_pipeline.predict(test_cases)
probabilities = best_pipeline.predict_proba(test_cases)[:, 1]

print("\n" + "=" * 60)
print("PREDICTION RESULTS")
print("=" * 60)

results_df = pd.DataFrame({
    'Customer': [f'Customer {i+1}' for i in range(len(test_cases))],
    'Tenure (months)': test_cases['tenure'],
    'Contract': test_cases['Contract'],
    'Monthly Charges': test_cases['MonthlyCharges'],
    'Churn Probability': [f'{p*100:.1f}%' for p in probabilities],
    'Prediction': ['Churn' if p == 1 else 'No Churn' for p in predictions]
})

print("\n", results_df.to_string(index=False))

# Display detailed prediction for each customer
print("\n" + "=" * 60)
print("DETAILED PREDICTIONS")
print("=" * 60)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"\nCustomer {i+1}:")
    print(f"  Contract: {test_cases['Contract'].iloc[i]}")
    print(f"  Tenure: {test_cases['tenure'].iloc[i]} months")
    print(f"  Monthly Charges: ${test_cases['MonthlyCharges'].iloc[i]:.2f}")
    print(f"  Churn Probability: {prob*100:.1f}%")
    print(f"  Prediction: {'⚠️ WILL CHURN' if pred == 1 else '✅ WILL NOT CHURN'}")

edge_cases = pd.DataFrame({
    'gender': ['Male', 'Female', 'Male', 'Female'],
    'SeniorCitizen': [0, 0, 1, 1],
    'Partner': ['No', 'No', 'No', 'No'],
    'Dependents': ['No', 'No', 'No', 'No'],
    'tenure': [1, 2, 1, 3],  # New customers
    'PhoneService': ['Yes', 'Yes', 'Yes', 'Yes'],
    'MultipleLines': ['No', 'No', 'Yes', 'Yes'],
    'InternetService': ['Fiber optic', 'Fiber optic', 'Fiber optic', 'Fiber optic'],
    'OnlineSecurity': ['No', 'No', 'No', 'No'],
    'OnlineBackup': ['No', 'No', 'No', 'No'],
    'DeviceProtection': ['No', 'No', 'No', 'No'],
    'TechSupport': ['No', 'No', 'No', 'No'],
    'StreamingTV': ['Yes', 'Yes', 'Yes', 'Yes'],
    'StreamingMovies': ['Yes', 'Yes', 'Yes', 'Yes'],
    'Contract': ['Month-to-month', 'Month-to-month', 'Month-to-month', 'Month-to-month'],  # High risk
    'PaperlessBilling': ['Yes', 'Yes', 'Yes', 'Yes'],
    'PaymentMethod': ['Electronic check', 'Electronic check', 'Electronic check', 'Electronic check'],  # High churn risk
    'MonthlyCharges': [120.00, 150.00, 130.00, 100.00],
    'TotalCharges': [120.00, 300.00, 130.00, 300.00]
})

# Make predictions
edge_predictions = best_pipeline.predict(edge_cases)
edge_probabilities = best_pipeline.predict_proba(edge_cases)[:, 1]

print("=" * 60)
print("EDGE CASE TEST - HIGH RISK CUSTOMERS")
print("(New customers, Month-to-month, Electronic check)")
print("=" * 60)

edge_results = pd.DataFrame({
    'Customer': [f'Risk Case {i+1}' for i in range(len(edge_cases))],
    'Tenure': edge_cases['tenure'],
    'Monthly Charges': edge_cases['MonthlyCharges'],
    'Churn Probability': [f'{p*100:.1f}%' for p in edge_probabilities],
    'Prediction': ['Churn' if p == 1 else 'No Churn' for p in edge_predictions]
})

print("\n", edge_results.to_string(index=False))

def predict_churn(customer_data, model_pipeline=best_pipeline):
    """
    Predict customer churn

    Parameters:
    -----------
    customer_data : DataFrame
        Customer information
    model_pipeline : Pipeline
        Trained model pipeline

    Returns:
    --------
    dict: Prediction results
    """
    prediction = model_pipeline.predict(customer_data)[0]
    probability = model_pipeline.predict_proba(customer_data)[0][1]

    return {
        'will_churn': bool(prediction),
        'churn_probability': probability,
        'churn_probability_percent': f'{probability*100:.1f}%',
        'risk_level': 'High' if probability > 0.7 else ('Medium' if probability > 0.4 else 'Low')
    }

# Example usage
print("=" * 60)
print("EXAMPLE: USING PREDICTION FUNCTION")
print("=" * 60)

# Single customer test
single_customer = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['Yes'],
    'tenure': [36],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['DSL'],
    'OnlineSecurity': ['Yes'],
    'OnlineBackup': ['Yes'],
    'DeviceProtection': ['Yes'],
    'TechSupport': ['Yes'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': ['Two year'],
    'PaperlessBilling': ['No'],
    'PaymentMethod': ['Credit card'],
    'MonthlyCharges': [70.00],
    'TotalCharges': [2520.00]
})

result = predict_churn(single_customer)

print("\nCustomer Profile:")
print("  - Gender: Male")
print("  - Senior Citizen: No")
print("  - Partner: Yes")
print("  - Dependents: Yes")
print("  - Tenure: 36 months")
print("  - Contract: Two year")
print("  - Internet: DSL")
print("  - Online Security: Yes")
print("\nPrediction Results:")
for key, value in result.items():
    print(f"  {key}: {value}")

def test_custom_prediction(model_pipeline):
    """
    Prompts the user for key customer metrics and returns a churn prediction
    using the trained model pipeline.
    """
    print("\n" + "="*50)
    print("           TEST MODEL WITH CUSTOM DATA     ")
    print("="*50)
    print("Enter customer details below (Press ENTER to keep the default value):")

    # 1. Base Default Customer Profile (All columns required by the pipeline)
    custom_data = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 12,              # Important Feature
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month', # Important Feature
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 75.00,      # Important Feature
        'TotalCharges': 900.00
    }

    # 2. Ask user for key valuable columns
    try:
        tenure_in = input(f"1. Tenure in months (Default: {custom_data['tenure']}): ")
        if tenure_in.strip():
            custom_data['tenure'] = int(tenure_in)

        monthly_in = input(f"2. Monthly Charges in $ (Default: {custom_data['MonthlyCharges']}): ")
        if monthly_in.strip():
            custom_data['MonthlyCharges'] = float(monthly_in)

        contract_in = input(f"3. Contract [Month-to-month, One year, Two year] (Default: {custom_data['Contract']}): ")
        if contract_in.strip() in ['Month-to-month', 'One year', 'Two year']:
            custom_data['Contract'] = contract_in

        # Dynamically adjust TotalCharges based on input to keep math realistic
        custom_data['TotalCharges'] = custom_data['tenure'] * custom_data['MonthlyCharges']

    except ValueError:
        print("\n[Error] Invalid input detected. Falling back to default numbers.")

    # 3. Convert dictionary to pandas DataFrame (1 row)
    import pandas as pd
    input_df = pd.DataFrame([custom_data])

    # 4. Make Prediction using the pipeline
    # The pipeline automatically scales the numbers and encodes the text features
    try:
        prediction = model_pipeline.predict(input_df)[0]
        probability = model_pipeline.predict_proba(input_df)[0][1]

        # 5. Display the result
        print("\n" + "-"*50)
        print("PREDICTION RESULT:")
        print("-"*50)

        if prediction == 1:
            print(f"⚠️ HIGH RISK: This customer is LIKELY TO CHURN.")
            print(f"Risk Probability: {probability:.2%}")
        else:
            print(f"✅ SAFE: This customer is LIKELY TO STAY (No Churn).")
            print(f"Churn Probability: {probability:.2%}")

        print("-"*50 + "\n")

    except Exception as e:
        print(f"\nAn error occurred during prediction. Please ensure the model pipeline is trained. Details: {e}")

# Run the interactive function using our best trained pipeline from Step 6
test_custom_prediction(best_pipeline)

# Final Summary
print("=" * 70)
print("                    FINAL SUMMARY - CUSTOMER CHURN PREDICTION")
print("=" * 70)

print("\n📊 DATASET OVERVIEW:")
print(f"   - Total Customers: {len(df)}")
print(f"   - Features: {len(X.columns)}")
print(f"   - Churn Rate: {df['Churn'].value_counts(normalize=True)[1]*100:.2f}%")

print("\n📈 MODEL COMPARISON (Test Set):")
print("-" * 70)
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
print("-" * 70)
for name, metrics in test_results.items():
    print(f"{name:<25} {metrics['Accuracy']:<12.4f} {metrics['Precision']:<12.4f} {metrics['Recall']:<12.4f} {metrics['F1 Score']:<12.4f} {metrics['ROC-AUC']:<12.4f}")
print("-" * 70)

print("\n🏆 BEST MODEL:", best_model_name)
print(f"   - Accuracy:  {test_results[best_model_name]['Accuracy']:.4f}")
print(f"   - Precision: {test_results[best_model_name]['Precision']:.4f}")
print(f"   - Recall:    {test_results[best_model_name]['Recall']:.4f}")
print(f"   - F1 Score:  {test_results[best_model_name]['F1 Score']:.4f}")
print(f"   - ROC-AUC:   {test_results[best_model_name]['ROC-AUC']:.4f}")

print("\n💡 KEY INSIGHTS:")
print("   1. Contract type is a strong predictor of churn")
print("   2. Month-to-month contracts have highest churn risk")
print("   3. Longer tenure reduces churn probability")
print("   4. Electronic check payment method associated with higher churn")
print("   5. Fiber optic internet customers have higher churn rates")

print("\n" + "=" * 70)
print("                                PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)

