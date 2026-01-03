import streamlit as st
import pandas as pd
import requests
import io
import joblib

# model
model = joblib.load("model.pkl")

# config
st.set_page_config(
    page_title="Price Sensitivity – Single Prediction",
    #page_icon="💸",
    layout="wide"
)

st.title("Price Optimization – Will the customer keep buying after a price increase?")
st.markdown(
    "Enter a single customer profile or Upload an Excel sheet to predict whether they will continue buying after a price increase." 
)


# menu
menu = st.sidebar.radio(
    "Select mode",
    ["1 Single Prediction", "2 Batch Prediction (Excel)"],
    format_func=lambda x: x.split(" ")[1]
)

# Helper: run prediction for a dataframe
FEATURE_COLS = [
    "total_spent",
    "avg_order_value",
    "avg_purchase_frequency",
    "days_since_last_purchase",
    "discount_behavior",
    "loyalty_program_member",
    "days_in_advance",
    "flight_type",
    "cabin_class",
]

def predict_df(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe with FEATURE_COLS and returns df with:
    - Probability
    - will_buy_after_price_increase (0/1)
    """
    X = df_input[FEATURE_COLS].copy()

    # Ensure dtypes are reasonable (avoid Excel weird types)
    numeric_cols = [
        "total_spent",
        "avg_order_value",
        "avg_purchase_frequency",
        "days_since_last_purchase",
        "discount_behavior",
        "loyalty_program_member",
        "days_in_advance",
    ]
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Simple safety: fill missing numeric with median of provided batch (or 0 if all missing)
    for c in numeric_cols:
        med = X[c].median()
        if pd.isna(med):
            med = 0
        X[c] = X[c].fillna(med)

    # Ensure categorical columns are strings
    X["flight_type"] = X["flight_type"].astype(str)
    X["cabin_class"] = X["cabin_class"].astype(str)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    out = df_input.copy()
    out["Probability"] = probs
    out["will_buy_after_price_increase"] = preds
    return out

# 1.Single Prediction
if menu.startswith("1"):
    st.header("Single-customer Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_spent = st.number_input("Total spent (USD)", min_value=0.0, value=3000.0, step=10.0)
        avg_order_value = st.number_input("Average order value (USD)", min_value=0.0, value=400.0, step=5.0)
        avg_purchase_frequency = st.number_input("Average purchase frequency (per month)", min_value=0.0, value=4.0, step=0.1)

    with col2:
        days_since_last_purchase = st.number_input("Days since last purchase", min_value=0, value=30, step=1)
        discount_behavior = st.number_input(
            "Discount behaviour (share of orders with coupon)",
            min_value=0.0, max_value=1.0, value=0.2, step=0.05, format="%.2f"
        )
        loyalty_program_member = st.selectbox(
            "Loyalty program member",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col3:
        days_in_advance = st.number_input("Days booked in advance", min_value=0, value=14, step=1)
        flight_type = st.selectbox("Flight type", options=["domestic", "international"])
        cabin_class = st.selectbox("Cabin class", options=["economy", "premium_economy", "business"])

    if st.button("Predict"):
        one = pd.DataFrame([{
            "total_spent": total_spent,
            "avg_order_value": avg_order_value,
            "avg_purchase_frequency": avg_purchase_frequency,
            "days_since_last_purchase": days_since_last_purchase,
            "discount_behavior": discount_behavior,
            "loyalty_program_member": loyalty_program_member,
            "days_in_advance": days_in_advance,
            "flight_type": flight_type,
            "cabin_class": cabin_class
        }])

        with st.spinner("Running local prediction…"):
            result = predict_df(one)

        prob = float(result.loc[0, "Probability"])
        pred = int(result.loc[0, "will_buy_after_price_increase"])

        label = "✅ Likely to **continue** buying" if pred == 1 else "⚠️ Likely to **stop** buying"
        st.markdown(f"### {label}\nProbability of continuing: **{prob:.1%}**")

# 2. Batch Prediction (Excel)
elif menu.startswith("2"):
    st.header("Batch Prediction – Upload Excel")

    st.markdown(
        "Your Excel file should contain these 9 columns:\n"
        f"`{', '.join(FEATURE_COLS)}`"
    )

    uploaded = st.file_uploader("Upload .xlsx file", type=["xlsx"])

    if uploaded is not None:
        df = pd.read_excel(uploaded)
        st.write("Preview of uploaded data:", df.head())

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns in Excel: {missing}")
            st.stop()

        if st.button("Run batch prediction"):
            with st.spinner("Running batch predictions locally…"):
                out_df = predict_df(df)

            st.success("Batch prediction complete.")
            st.write(out_df.head())

            # Download as Excel
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                out_df.to_excel(writer, index=False)
            towrite.seek(0)

            st.download_button(
                label="Download results as Excel",
                data=towrite,
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )