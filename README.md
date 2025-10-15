# 💳 Real-Time Credit Card Fraud Detection System

An end-to-end application that simulates real-time credit card fraud detection using a machine learning model served via a REST API with an interactive web interface.

## 🌟 Introduction

Credit card fraud is a major concern for financial institutions and customers alike. This project demonstrates a complete pipeline for building a fraud detection system, from training a machine learning model to deploying it with a user-friendly interface.

The application allows a user to simulate a credit card transaction. This data is sent to a backend API, which uses a pre-trained logistic regression model to predict the probability of the transaction being fraudulent in real-time.

## ✨ Key Features

* **🧠 Intelligent Fraud Detection**: Utilizes a Scikit-learn model trained on a Kaggle dataset with over 280,000 transactions.
* **⚡️ Real-Time Predictions**: A robust FastAPI backend provides immediate fraud analysis for each transaction.
* **🖥️ Interactive UI**: A clean and simple user interface built with Streamlit for simulating transactions and viewing results.
* **📂 Transaction History**: All simulated transactions and their predictions are logged and can be viewed in the UI.
* **📦 Modular & Scalable**: A decoupled architecture (ML model, backend, frontend) makes the system easy to maintain and extend.

## 🚀 Tech Stack

* **Backend**: FastAPI, Uvicorn
* **Frontend**: Streamlit
* **Machine Learning**: Scikit-learn, Pandas, Joblib
* **Core Language**: Python

## 🛠️ Getting Started

Follow these steps to get the full application running on your local machine.

### Prerequisites

* Python 3.8 or higher
* Git

### Installation & Running the App

1.  **Clone the Repository**
2.  **Create and Activate a Virtual Environment**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    # Create the environment
    python3 -m venv venv
    
    # Activate it
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Required Packages**
    ```bash
    pip install pandas scikit-learn joblib requests fastapi uvicorn streamlit
    ```

4.  **Train the Machine Learning Model**
    This script will download the dataset, train the model, and save the necessary files (`.joblib`) in the `ml/model/` directory.
    ```bash
    python3 ml/train_model.py
    ```

5.  **Run the Backend API**
    With the model trained, start the FastAPI server.
    ```bash
    uvicorn backend.main:app --reload
    ```

6.  **Run the Frontend Application**
    Open a **new terminal** (and activate the virtual environment again).
    ```bash
    streamlit run frontend/app.py
    ```

## 🏗️ How It Works

1.  **Frontend (Streamlit)**: The user enters transaction details into the web interface and submits them.
2.  **Backend (FastAPI)**: The frontend sends a `POST` request with the transaction data to the backend API. The backend preprocesses this data and passes it to the ML model.
3.  **Machine Learning Model**: The trained logistic regression model predicts whether the transaction is fraudulent (`1`) or genuine (`0`) and calculates the fraud probability.
4.  **Response**: The API sends the prediction result back to the frontend, which displays it to the user in a clear, color-coded message.
