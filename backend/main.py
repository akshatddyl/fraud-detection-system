from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import List

# --- App Initialization ---
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An API to detect fraudulent credit card transactions.",
    version="1.0.0"
)

# --- Load Model and Scaler ---
# Use relative paths to ensure it works in different environments
try:
    model = joblib.load('ml/model/fraud_detection_model.joblib')
    scaler = joblib.load('ml/model/scaler.joblib')
except FileNotFoundError:
    raise RuntimeError("Model or scaler not found. Please run the 'ml/train_model.py' script first.")


# --- Data Models ---
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class PredictionResponse(BaseModel):
    is_fraud: int
    probability_fraud: float
    probability_genuine: float
    
class TransactionRecord(Transaction):
    prediction: PredictionResponse


# --- Local Database (CSV File) ---
DB_FILE = "backend/transaction_history.csv"

def initialize_db():
    """Creates the CSV file with headers if it doesn't exist."""
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=[*Transaction.model_fields.keys(), 'is_fraud', 'probability_fraud'])
        df.to_csv(DB_FILE, index=False)

initialize_db()

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to welcome users to the API."""
    return {"message": "Welcome to the Fraud Detection API. Go to /docs for documentation."}

@app.post("/predict/", response_model=PredictionResponse, tags=["Prediction"])
def predict_fraud(transaction: Transaction):
    """
    Predicts if a transaction is fraudulent.
    Receives transaction data, preprocesses it, and returns the fraud prediction.
    """
    try:
        # Create a DataFrame from the input transaction
        df = pd.DataFrame([transaction.model_dump()])

        # Preprocess: Scale 'Time' and 'Amount' using the loaded scaler
        # We need to reshape for a single sample
        df['scaled_amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = scaler.transform(df['Time'].values.reshape(-1, 1))
        
        # Drop original columns and select features for the model
        df_processed = df.drop(['Time', 'Amount'], axis=1)
        
        # Reorder columns to match the training order (important!)
        # The model expects columns in the same order as during training.
        # This is a robust way to ensure the order is correct.
        training_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                         'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                         'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                         'scaled_amount', 'scaled_time']
        df_processed = df_processed[training_cols]


        # Make prediction
        prediction = model.predict(df_processed)[0]
        probabilities = model.predict_proba(df_processed)[0]
        
        response = {
            "is_fraud": int(prediction),
            "probability_fraud": float(probabilities[1]),
            "probability_genuine": float(probabilities[0])
        }

        # Save the transaction and its prediction to the history
        transaction_record = transaction.model_dump()
        transaction_record['is_fraud'] = response['is_fraud']
        transaction_record['probability_fraud'] = response['probability_fraud']
        
        history_df = pd.DataFrame([transaction_record])
        history_df.to_csv(DB_FILE, mode='a', header=False, index=False)
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/", response_model=List[dict], tags=["History"])
def get_transaction_history():
    """
    Retrieves the list of all transactions and their predictions.
    """
    if not os.path.exists(DB_FILE):
        return []
    try:
        df = pd.read_csv(DB_FILE)
        return df.to_dict('records')
    except pd.errors.EmptyDataError:
        return [] # Return empty list if the csv is empty
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
