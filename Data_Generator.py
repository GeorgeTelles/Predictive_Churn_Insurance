import pandas as pd
import numpy as np
import random
from random import choices

np.random.seed(42)
random.seed(42)

n_customers = 30000

data = {
    'customer_id': [],
    'age': [],
    'gender': [],
    'location': [],
    'customer_tenure_years': [],
    'policy_type': [],
    'monthly_premium': [],
    'coverage_level': [],
    'claims_last_year': [],
    'payment_delay_days': [],
    'customer_satisfaction': [],
    'customer_service_interactions': [],
    'churn': []
}

# Data population
for id in range(1, n_customers + 1):
    # Basic data
    data['customer_id'].append(id)
    data['age'].append(int(np.random.normal(45, 10)))
    data['gender'].append(choices(['M', 'F'], weights=[0.55, 0.45])[0])
    data['location'].append(choices(['Urban', 'Suburban', 'Rural'], weights=[0.6, 0.3, 0.1])[0])
    
    # Customer tenure
    data['customer_tenure_years'].append(round(np.random.exponential(scale=5)) + 1)
    
    # Policy type
    policy_type = choices(['Comprehensive', 'Third_Party', 'Basic'], weights=[0.4, 0.4, 0.2])[0]
    data['policy_type'].append(policy_type)
    
    # Monthly premium calculation
    base_premium = 100 + (data['age'][-1] * 0.5)
    if policy_type == 'Comprehensive':
        base_premium *= 2.5
    elif policy_type == 'Third_Party':
        base_premium *= 1.8
    data['monthly_premium'].append(round(base_premium + np.random.normal(0, 20), 2))
    
    # Coverage level
    data['coverage_level'].append(choices(['High', 'Medium', 'Low'], 
                                      weights=[0.3, 0.5, 0.2])[0])
    
    # Claims in last year
    data['claims_last_year'].append(np.random.poisson(0.5))
    
    # Payment delays
    if np.random.rand() < 0.7: 
        data['payment_delay_days'].append(0)
    else:
        data['payment_delay_days'].append(np.random.poisson(15))
    
    # Customer satisfaction (1-5)
    data['customer_satisfaction'].append(choices([1, 2, 3, 4, 5], 
                                              weights=[0.1, 0.2, 0.3, 0.25, 0.15])[0])
    
    data['customer_service_interactions'].append(np.random.poisson(2))
    
    # Churn probability calculation
    churn_score = (
        -0.1 * data['age'][-1] +
        0.3 * (data['monthly_premium'][-1]/100) +
        0.4 * data['payment_delay_days'][-1] +
        -0.5 * data['customer_satisfaction'][-1] +
        -0.2 * data['customer_tenure_years'][-1] +
        0.3 * data['claims_last_year'][-1] +
        0.2 * data['customer_service_interactions'][-1] +
        np.random.normal(0, 0.5)
    )
    
    # Convert to probability using logistic function
    churn_prob = 1 / (1 + np.exp(-churn_score))
    data['churn'].append(1 if churn_prob > 0.5 else 0)

df = pd.DataFrame(data)

# Handle extreme values
df['age'] = df['age'].clip(18, 80)
df['monthly_premium'] = df['monthly_premium'].clip(100, 600)
df['payment_delay_days'] = df['payment_delay_days'].clip(0, 60)

# Save to CSV
df.to_csv('customer_data.csv', index=False)

print("Historical customer database created successfully!")
print(f"Dataset size: {df.shape}")
print(f"Churn rate: {df['churn'].mean():.2%}")