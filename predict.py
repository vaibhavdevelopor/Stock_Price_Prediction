import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import base64
import gdown
import os
from tensorflow.keras.models import load_model

# Google Drive file ID
file_id = "1x-MOTB6byjoAKmRPLnR0SkU577e9IciS"

# Model path
model_path = "Stock_Predictions_Model.keras"

# Check if the model exists, if not, download it
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Function to set background image
def set_background(image_path):
    """Sets the background image for Streamlit app."""
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    bg_image_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(bg_image_style, unsafe_allow_html=True)

# Google Drive Image File ID
image_file_id = "17Hf_WfHm76AnDVI9THkLg1QHZ2SK-R3H"
image_path = "background.jpg"

# Check if the image exists, if not, download it
if not os.path.exists(image_path):
    try:
        url = f"https://drive.google.com/uc?export=download&id={image_file_id}"
        gdown.download(url, image_path, quiet=False)
    except Exception as e:
        st.error(f"Image download failed: {e}")

# Set background image
set_background(image_path)

# Streamlit UI
st.title("📈 Stock Market Price Prediction")
st.write("Predict stock prices using a trained LSTM model.")

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol (e.g., AAPL, TSLA, GOOG):', 'GOOG')

start = '2012-01-01'
end = '2025-03-14'

# Fetch Stock Data
if stock:
    data = yf.download(stock, start=start, end=end)

    # Splitting data into training and testing
    data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
    data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

    # Normalize Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.transform(data_test)

    # Calculate Moving Averages
    ma_50_days = data['Close'].rolling(50).mean().dropna()
    ma_100_days = data['Close'].rolling(100).mean().dropna()
    ma_200_days = data['Close'].rolling(200).mean().dropna()

    # Plot: Price vs 50-day MA
    st.subheader('Stock Price vs 50-day Moving Average')
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r', label="50-day MA")
    plt.plot(data['Close'], 'g', label="Stock Price")
    plt.legend()
    st.pyplot(fig1)

    # Plot: Price vs 50-day & 100-day MA
    st.subheader('Stock Price vs 50-day & 100-day Moving Averages')
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r', label="50-day MA")
    plt.plot(ma_100_days, 'b', label="100-day MA")
    plt.plot(data['Close'], 'g', label="Stock Price")
    plt.legend()
    st.pyplot(fig2)

    # Plot: Price vs 100-day & 200-day MA
    st.subheader('Stock Price vs 100-day & 200-day Moving Averages')
    fig3 = plt.figure(figsize=(8, 6))
    plt.plot(ma_100_days, 'r', label="100-day MA")
    plt.plot(ma_200_days, 'b', label="200-day MA")
    plt.plot(data['Close'], 'g', label="Stock Price")
    plt.legend()
    st.pyplot(fig3)

    # Prepare data for LSTM model
    x_test, y_test = [], []
    past_100_days = data_train.tail(100)
    data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scaled = scaler.transform(data_test_full)

    for i in range(100, data_test_scaled.shape[0]):
        x_test.append(data_test_scaled[i - 100:i])
        y_test.append(data_test_scaled[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Make Predictions
    predictions = model.predict(x_test)

    # Rescale Predictions
    scale_factor = 1 / scaler.scale_[0]
    predictions = predictions * scale_factor
    y_test = y_test * scale_factor

    # Plot: Actual vs Predicted Prices
    st.subheader('Actual Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8, 6))
    plt.plot(predictions, 'r', label='Predicted Price')
    plt.plot(y_test, 'g', label='Actual Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig4)
