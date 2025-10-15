import streamlit as st
import requests
import pandas as pd
import random
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --- API URL ---
# Assumes the FastAPI backend is running on the default port 8000
API_URL = "http://127.0.0.1:8000"

# --- Page Title and Description ---
st.title("💳 Real-Time Credit Card Fraud Detection")
st.markdown("""
This application simulates a real-time credit card fraud detection system. 
Enter transaction details below or use the auto-fill button to generate a sample transaction. 
The system will predict whether the transaction is genuine or fraudulent.
""")

# --- Helper Functions ---
def get_history():
    """Fetches transaction history from the backend."""
    try:
        response = requests.get(f"{API_URL}/history/")
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            st.error(f"Failed to fetch history. Status code: {response.status_code}")
            return pd.DataFrame()
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the backend API. Please ensure the FastAPI server is running.")
        return pd.DataFrame()

# --- Main Application ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Simulate a Transaction")

    with st.form("transaction_form"):
        # We only need a few fields for simulation. The rest will be randomized.
        amount = st.number_input("Amount", min_value=0.01, max_value=1000000.0, value=150.75, step=10.0)
        
        # Use a button to auto-fill with random (but plausible) data
        if st.form_submit_button("Auto-fill with Sample Data"):
            # Generate random but plausible values for the 28 PCA components
            v_features = {f'V{i}': random.uniform(-5, 5) for i in range(1, 29)}
        else:
            v_features = {f'V{i}': 0.0 for i in range(1, 29)} # Default to 0 if not auto-filled
            
        submitted = st.form_submit_button("Submit Transaction")

        if submitted:
            # Construct the transaction data payload
            transaction_data = {
                "Time": time.time() % 172800,  # Use seconds since epoch, modulo 2 days
                "Amount": amount,
                **v_features
            }
            
            with st.spinner('Analyzing transaction...'):
                try:
                    # Send data to the FastAPI backend
                    response = requests.post(f"{API_URL}/predict/", json=transaction_data)
                    
                    if response.status_code == 200:
                        prediction = response.json()
                        
                        is_fraud = prediction['is_fraud']
                        prob_fraud = prediction['probability_fraud'] * 100
                        
                        if is_fraud == 1:
                            st.error(f"**FRAUD DETECTED!** (Probability: {prob_fraud:.2f}%)")
                            st.image("https://i.imgur.com/Qk7a52f.png", width=100) # Simple alert icon
                        else:
                            st.success(f"**Transaction is Genuine** (Fraud Probability: {prob_fraud:.2f}%)")
                    else:
                        st.error(f"Error from API: {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("Connection Error: Could not connect to the backend API.")


with col2:
    st.header("Transaction History")
    
    if st.button("Refresh History"):
        history_df = get_history()
        if not history_df.empty:
            # Display history with the most recent transactions first
            st.dataframe(history_df.sort_index(ascending=False), use_container_width=True)
        else:
            st.info("No transaction history found.")
    else:
        # Load history on first run
        history_df = get_history()
        if not history_df.empty:
            st.dataframe(history_df.sort_index(ascending=False), use_container_width=True)
        else:
            st.info("No transaction history found.")
