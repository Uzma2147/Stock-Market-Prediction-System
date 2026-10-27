Stock Market Prediction System

A Machine Learning and Data Science project developed as part of the Data Science Internship program at Sqrock IT Solutions (Project Phase 2).

This application collects historical stock market data, performs data preprocessing and feature engineering, trains a machine learning model, and displays predictions via an interactive Streamlit web application.

 Key Features

Automated Data Collection: Downloads historical stock prices (Open, High, Low, Close, Volume) dynamically using Yahoo Finance (yfinance).

Feature Engineering: Computes technical indicators including 50-day SMA and 200-day SMA (Simple Moving Averages) to capture market trends.

Machine Learning Model: Utilizes a Random Forest Regressor model trained on time-series split data to forecast next-day closing prices.

Model Evaluation Metrics: Calculates standard performance indicators including MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R² Score.

Interactive Web Dashboard: Built with Streamlit to allow users to select ticker symbols, adjust date ranges, view raw data, and visualize actual vs. predicted prices in real time.

 Tech Stack & Dependencies

Language: Python 3.8+

Web Framework: Streamlit

Data Processing: Pandas, NumPy

Data Fetching: Yahoo Finance (yfinance)

Machine Learning: Scikit-Learn

Visualization: Matplotlib

Model Persistence: Joblib




-> Local Setup & Execution Instructions


2. Install Dependencies

Make sure you have Python installed. Run the following command in your terminal:

pip install -r requirements.txt


3. Launch the Streamlit Web Application

Run the Streamlit server using Python:

python -m streamlit run app.py


-> Performance Evaluation

The model uses historical time-series split data (80% training, 20% testing) to ensure chronological integrity without data leakage. Model performance is continuously computed and displayed on the Streamlit dashboard:

Mean Absolute Error (MAE): Measures average absolute deviation of predicted prices.

Root Mean Squared Error (RMSE): Penalizes larger prediction errors heavily.

R² Score: Indicates the proportion of variance explained by the model.

 -> License & Acknowledgments


Open-source libraries used: Streamlit, Scikit-Learn, yfinance, Pandas, Matplotlib.
