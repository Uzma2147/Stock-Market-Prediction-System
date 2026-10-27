import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page Configuration
st.set_page_config(page_title="Stock Market Prediction System", layout="wide")

st.title(" Stock Market Prediction Dashboard")
st.write("An interactive web application to analyze historical stock data and predict future prices using Machine Learning.")

# Sidebar User Inputs
st.sidebar.header("Configuration Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2026-01-01"))

@st.cache_data
def load_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

# Load data with a spinner animation
with st.spinner(f"Fetching data for {ticker}..."):
    data = load_stock_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data found for the selected ticker or date range. Please try another symbol.")
else:
    # Display raw historical data table
    st.subheader(f"Historical Stock Data: {ticker}")
    st.dataframe(data.tail(10), use_container_width=True)

    # Feature Engineering
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Target'] = data['Close'].shift(-1)
    data_clean = data.dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200']
    X = data_clean[features]
    y = data_clean['Target']

    # Chronological Train-Test Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance Metrics Cards
    st.subheader("Model Performance Evaluation")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
    col3.metric("R2 Score", f"{r2:.2f}")

    # Visualization Plot
    st.subheader("Actual vs. Predicted Stock Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.index, y_test.values, label='Actual Price', color='blue', alpha=0.8)
    ax.plot(y_test.index, y_pred, label='Predicted Price', color='orange', linestyle='--', alpha=0.9)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.set_title(f"{ticker} - Actual vs Predicted Closing Prices", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)