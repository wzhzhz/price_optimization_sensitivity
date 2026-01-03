import pandas as pd
import numpy as np

np.random.seed(42)

n_customers = 50

data = {
    "total_spent": np.round(np.random.uniform(200, 15000, n_customers), 2),
    "avg_order_value": np.round(np.random.uniform(50, 1200, n_customers), 2),
    "avg_purchase_frequency": np.round(np.random.uniform(0.5, 12, n_customers), 2),
    "days_since_last_purchase": np.random.randint(0, 365, n_customers),
    "discount_behavior": np.round(np.random.uniform(0.0, 1.0, n_customers), 2),
    "loyalty_program_member": np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
    "days_in_advance": np.random.randint(0, 180, n_customers),
    "flight_type": np.random.choice(
        ["domestic", "international"],
        n_customers,
        p=[0.7, 0.3]
    ),
    "cabin_class": np.random.choice(
        ["economy", "premium_economy", "business"],
        n_customers,
        p=[0.65, 0.2, 0.15]
    ),
}

df = pd.DataFrame(data)

# save as Excel
output_path = "test_50_customers.xlsx"
df.to_excel(output_path, index=False)

#print(f"Saved test file: {output_path}")
#print(df.head())
