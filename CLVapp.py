import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the model with caching
@st.cache_resource()
def load_model():
    return joblib.load('RF_Regressor.pkl')

st.cache_resource.clear()

# Website title and subtitle
st.title('Customer Lifetime Value Prediction Tool')
st.header('This tool helps TrendStyles prioritize customer retention efforts.')

# Load the trained model
model = load_model()

# User Input Form
if model:
    st.subheader('Please enter the following details:')

#Review Rating
Review_Rating = st.number_input(
    "Review Rating",
    min_value=1.0, max_value=5.0, value=1.0,
    help="Enter a value between 1.0 and 5.0"
)

# Discount Applied
Discount_Applied = st.selectbox(
    "Discount Applied",
    options=[(0, 'No'), (1, 'Yes')],
    format_func=lambda x: x[1],
    help="Select if discount was applied"
)
Discount_Applied_Value = Discount_Applied[0]

#Promo Code Used
Promo_Code_Used= st.selectbox(
    "Promo Code Used",
    options=[(0, 'No'), (1, 'Yes')],
    format_func=lambda x: x[1],
    help="Select if promo code was used was applied"
)
Promo_Code_Used_Value = Promo_Code_Used[0]

#Churn
Churn= st.selectbox(
    "Churn",
    options=[(0, 'No'), (1, 'Yes')],
    format_func=lambda x: x[1],
    help="Select if customer churned"
)
Churn_Value = Churn[0]

# Purchase amount
Purchase_Amount_USD = st.number_input(
    "Purchase amount",
    min_value=20, max_value=100, value=20,
    help="Enter a value in USD"
)

# Previous Purchases
Previous_Purchases = st.number_input(
    "Previous Purchases",
    min_value=1, max_value=50, value=1,
    help="Enter a value between 1 and 50"
)

# Prediction
if st.button("Predict") and model:
    user_input = np.array([Review_Rating, Discount_Applied_Value, Promo_Code_Used_Value, Churn_Value, Purchase_Amount_USD, Previous_Purchases])
    reshaped_input = user_input.reshape(1, -1)
    
    st.write("📊 **User Input:**")
    st.json({
        "Review Rating": Review_Rating,
        "Discount Applied": "Yes" if Discount_Applied_Value else "No",
        "Promo Code Used": "Yes" if Promo_Code_Used_Value else "No",
        "Churn": "Yes" if Churn_Value else "No",
        "Purchase Amount (USD)": Purchase_Amount_USD,
        "Previous Purchases": Previous_Purchases
    })
    
    prediction = model.predict(reshaped_input)
    
    st.subheader(f'📈 The predicted Customer Lifetime Value is **${prediction[0]:.3f}**')
elif not model:
    st.warning("⚠ Model not loaded. Please check the file `RF_Regressor.pkl`.")
