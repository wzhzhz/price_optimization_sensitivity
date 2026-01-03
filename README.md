# Price Optimization for Travel and E-commerce Platforms 
## Customer Continuation Prediction After Price Increase

## 1. Objective

The objective of this project is to predict whether a customer will continue purchasing after a price increase, and to estimate the probability of continuation.

The model is designed to support revenue and pricing teams at travel or e-commerce platforms (e.g. flight booking websites) by helping them understand customer price sensitivity at an individual level.

This project demonstrates an end-to-end analytics workflow, including:
- Synthetic customer data generation
- Model training using logistic regression
- Probability-based customer continuation prediction
- A Streamlit app for single-customer and batch (Excel) inference



## 2. Features Used

The model uses the following customer-level features:

- `total_spent` – Total historical spend on the platform  
- `avg_order_value` – Average value per order  
- `avg_purchase_frequency` – Average number of purchases per month 
- `days_since_last_purchase` – Number of days since the last purchase  
- `discount_behavior` – Share of orders using discounts  
- `loyalty_program_member` – Loyalty program membership indicator (0 = No, 1 = Yes)
- `days_in_advance` – Number of days bookings are made in advance
- `flight_type` – Domestic or international flight
- `cabin_class` – Economy, premium economy, or business  

The target variable is:

- `will_buy_after_price_increase` – Binary outcome indicating whether the customer continues buying after a price increase (0 / 1)

## 3. Project Structure

```text

├── Data/
│   ├── data_simulation.py        # Synthetic data generation script
│   ├── test_data.py              # Script to generate random test data
│   └── test_50_customers.xlsx    # Example Excel file for batch prediction
│
├── Code/
│   ├── model_training.ipynb      # Model training and evaluation notebook
│   ├── model.pkl                 # Trained price sensitivity model
│   └── streamlit.py              # Streamlit app for single & batch prediction
│
├── .gitignore
├── README.md
└── requirements.txt
```


## 4. Model

- Algorithm: **Logistic Regression**
- Preprocessing:
  - StandardScaler for numeric features
  - One-hot encoding for categorical features
- Output:
  - Probability of continuing after a price increase
  - Binary classification (continue / stop)
- The model is trained and saved as a file (model.pkl) for reuse in the Streamlit.py.


## 5. How to Run the Streamlit App

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the application by typing the following command in the terminal
```bash
streamlit run streamlit.py
```
The application will open automatically in your browser.

## 6. Streamlit App Preview

<div align="center">
  <img
    src="https://github.com/user-attachments/assets/28042914-9802-4321-87cb-2e932b07c0e7"
    width="1000"
    height="1200"
    alt="Streamlit App Screenshot"
  />
</div>

[Click here to directly open the app](http://localhost:8502/)



## 7. Notes
The Streamlit application assumes that model.pkl is located in the same directory as Streamlit.py.
If the model file is moved to a different folder, the loading path in the Streamlit script must be updated accordingly.








