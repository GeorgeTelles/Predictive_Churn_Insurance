import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Exploratory Data Analysis
def perform_eda(df):
    print("\n=== Exploratory Data Analysis ===")
    
    # Churn Distribution
    print("\nChurn Distribution:")
    print(df['churn'].value_counts(normalize=True))
    
    # Correlation Analysis
    print("\nCorrelations with Churn:")
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()['churn'].sort_values(ascending=False)
    print(correlation)


# 2. Data Preprocessing
def preprocess_data(df):
    # Encoding Categorical Variables
    le = LabelEncoder()
    categorical_cols = ['gender', 'location', 'policy_type', 'coverage_level']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Feature Selection
    features = ['age', 'gender', 'location', 'customer_tenure_years', 
                'policy_type', 'monthly_premium', 'coverage_level',
                'claims_last_year', 'payment_delay_days',
                'customer_satisfaction', 'customer_service_interactions']
    
    X = df[features]
    y = df['churn']
    
    # Missing Value Imputation
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, features

# 3. Predictive Modeling
def train_model(X, y):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model Initialization
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Model Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Model Performance ===")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.2f}")
    
    return model

# 4. High-Risk Customer Identification
def identify_high_risk(model, df, features):
    # Preprocess Data
    X, _, _ = preprocess_data(df)
    
    # Calculate Churn Probabilities
    probabilities = model.predict_proba(X)[:, 1]
    df['churn_probability'] = probabilities
    
    # Identify Top 10 High-Risk Customers
    high_risk = df.sort_values('churn_probability', ascending=False).head(10)
    
    # Feature Importance Analysis
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n=== Key Drivers of Churn ===")
    print(importance_df)
    
    # Risk Explanation
    print("\n=== Customers at Highest Risk of Churn ===")
    for idx, row in high_risk.iterrows():
        print(f"\nCustomer ID: {row['customer_id']}")
        print(f"Churn Probability: {row['churn_probability']:.2%}")
        print("Key Risk Factors:")
        
        # Comparative Analysis
        if row['monthly_premium'] > df['monthly_premium'].median():
            print(f"- Monthly premium above average ({row['monthly_premium']:.2f} vs {df['monthly_premium'].median():.2f})")
        if row['customer_satisfaction'] < 3:
            print(f"- Low satisfaction ({row['customer_satisfaction']} vs average {df['customer_satisfaction'].mean():.1f})")
        if row['payment_delay_days'] > 0:
            print(f"- Payment delay ({row['payment_delay_days']} days)")
        if row['claims_last_year'] > 0:
            print(f"- Recent claims ({row['claims_last_year']})")
        if row['customer_service_interactions'] > df['customer_service_interactions'].median():
            print(f"- High service interactions ({row['customer_service_interactions']})")


if __name__ == "__main__":
    # Load Translated Dataset
    df = pd.read_csv('customer_data.csv')
    
    perform_eda(df)
    X, y, features = preprocess_data(df)
    model = train_model(X, y)
    identify_high_risk(model, df, features)