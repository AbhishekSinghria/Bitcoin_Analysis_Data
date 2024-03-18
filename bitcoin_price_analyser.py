import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def fetch_bitcoin_price_history(days=30, currency='usd', interval='daily'):
    try:
        # Define the API endpoint
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

        # Define the parameters for the API request
        params = {
            "vs_currency": currency,
            "days": str(days),
            "interval": interval
        }

        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for unsuccessful status codes

        # Convert the response to JSON format
        data = response.json()

        # Extract the prices from the response data
        prices = data['prices']

        # Create a DataFrame from the prices data
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])

        # Convert the timestamp to datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df

    except Exception as e:
        print("Error:", e)
        return None

def plot_bitcoin_price(df):
    if df is not None:
        # Plot the Bitcoin price over time
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['price'], marker='o', linestyle='-', color='white', label='Actual Price')
        plt.title('Bitcoin Price History')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.gca().set_facecolor('black')  # Set background color to black
        plt.show()

def forecast_bitcoin_price(df, forecast_days=7):
    if df is not None:
        # Feature engineering: Extract timestamp features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month

        # Split the data into training and testing sets
        X = df[['day_of_week', 'day_of_month', 'month']]
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions for future dates
        future_dates = pd.date_range(start=df['timestamp'].max(), periods=forecast_days + 1, freq='D')
        future_features = pd.DataFrame({
            'timestamp': future_dates,
            'day_of_week': future_dates.dayofweek,
            'day_of_month': future_dates.day,
            'month': future_dates.month
        })
        future_predictions = model.predict(future_features[['day_of_week', 'day_of_month', 'month']])

        # Plot the forecasted prices
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['price'], marker='o', linestyle='-', color='white', label='Actual Price')
        plt.plot(future_dates[1:], future_predictions[1:], marker='o', linestyle='--', color='green', label='Forecasted Price (Profit)')
        plt.title('Bitcoin Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.gca().set_facecolor('black')  # Set background color to black
        plt.show()

# Fetch Bitcoin price history
bitcoin_price_data = fetch_bitcoin_price_history()

# Plot Bitcoin price history
plot_bitcoin_price(bitcoin_price_data)

# Forecast Bitcoin price
forecast_bitcoin_price(bitcoin_price_data)
