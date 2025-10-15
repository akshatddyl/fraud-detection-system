import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import requests
import zipfile
import io

# --- 1. Data Download and Preparation ---

# Create a directory to store the model and data
if not os.path.exists('ml/data'):
    os.makedirs('ml/data')
if not os.path.exists('ml/model'):
    os.makedirs('ml/model')

# Download and extract the dataset from Kaggle
# Note: This is a simplified way. For automation, the Kaggle API would be better.
# For this script, we'll download it directly from a source that doesn't require login.
# A copy of the dataset is often hosted on other platforms for educational purposes.
# IMPORTANT: The original Kaggle link requires login. We use an alternative public source.
print("Downloading dataset...")
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
try:
    response = requests.get(url)
    response.raise_for_status() # Raise an exception for bad status codes
    df = pd.read_csv(io.BytesIO(response.content))
    df.to_csv('ml/data/creditcard.csv', index=False)
    print("Dataset downloaded and saved successfully.")
except requests.exceptions.RequestException as e:
    print(f"Error downloading the dataset: {e}")
    # As a fallback, try to load a local copy if it exists
    if os.path.exists('ml/data/creditcard.csv'):
        print("Loading local dataset.")
        df = pd.read_csv('ml/data/creditcard.csv')
    else:
        print("Could not download or find a local copy of the dataset. Please download it manually from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data and place it in the 'ml/data' directory.")
        exit()


# --- 2. Data Preprocessing and Feature Engineering ---

print("Preprocessing data...")

# The dataset is already well-preprocessed (PCA-transformed features)
# We just need to scale the 'Time' and 'Amount' columns.
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop the original 'Time' and 'Amount' columns, and the unused 'V' columns for this simple model
df = df.drop(['Time', 'Amount'], axis=1)

# --- 3. Model Training ---

print("Training model...")

# Define features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
# The dataset is highly imbalanced, so we use stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the Logistic Regression model
# We use class_weight='balanced' to handle the imbalanced dataset
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# --- 4. Model Evaluation ---

print("Evaluating model...")
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --- 5. Save the Trained Model ---

print("Saving the trained model...")
model_path = 'ml/model/fraud_detection_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Also save the scaler for the two columns
scaler_path = 'ml/model/scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

print("\n--- ML Pipeline Complete ---")
