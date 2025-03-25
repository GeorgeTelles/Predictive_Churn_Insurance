# Predictive Churn Model with High-Risk Customer Identification for Insurance  

This code implements an end-to-end solution to analyze and predict customer churn (cancellation) in the insurance industry using machine learning. The workflow includes data exploration, preprocessing, modeling, and actionable insights generation.

---

## **Project Stages**  

### 1. **Exploratory Data Analysis (EDA)**  
- Generates statistics on the distribution of `churn` in the dataset.  
- Calculates correlations between numerical variables and churn rates.  
- Identifies initial patterns to guide modeling decisions.  

### 2. **Data Preprocessing**  
- **Encoding:** Converts categorical variables (`gender`, `location`, `policy_type`, `coverage_level`) to numerical values using `LabelEncoder`.  
- **Feature Selection:** Defines predictor variables such as:  
  - `age` (idade)  
  - `gender` (genero)  
  - `location` (localizacao)  
  - `customer_tenure_years` (tempo_cliente_anos)  
  - `policy_type` (tipo_politica)  
  - `monthly_premium` (premio_mensal)  
  - `coverage_level` (nivel_cobertura)  
  - `claims_last_year` (sinistros_ultimo_ano)  
  - `payment_delay_days` (atraso_pagamento_dias)  
  - `customer_satisfaction` (satisfacao_cliente)  
  - `customer_service_interactions` (numero_interacoes_atendimento).  
- **Data Handling:**  
  - Imputes missing values with the median (`SimpleImputer`).  
  - Normalizes data using `StandardScaler`.  

### 3. **Predictive Modeling**  
- **Algorithm:** Random Forest with adjustments for class imbalance (`class_weight='balanced'`).  
- **Evaluation:**  
  - Metrics: Precision, recall, F1-score, confusion matrix, and AUC-ROC.  
  - Strategy: Stratified data split (80% training / 20% testing).  

### 4. **High-Risk Customer Identification**  
- **Classification:** Lists the top 10 customers with the highest churn probability.  
- **Explainability:**  
  - Highlights critical drivers (e.g., `monthly_premium` above average, `payment_delay_days`).  
  - Displays feature importance to explain model decisions.  

---

## **Technologies and Methods**  
- **Libraries:**  
  - `Pandas` and `NumPy` for data manipulation.  
  - `Scikit-learn` for machine learning and preprocessing.  
  - `Seaborn`/`Matplotlib` for visualization (implied by imports).  
- **Techniques:**  
  - Handling imbalanced data.  
  - Model interpretability (rule-based explanations).  

---

## **Model Output**  
1. **Performance Report:**  
   - Detailed classification metrics (precision, recall, F1).  
   - AUC-ROC score to evaluate model discrimination capability.  

2. **Priority Action List:**  
   - High-risk customers flagged for intervention, with specific reasons (e.g., low satisfaction, recent claims history).  

---

**Ideal for:** Insurance companies aiming to reduce customer churn through data-driven strategies and machine learning.  
