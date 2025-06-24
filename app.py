from flask import Flask, jsonify
import yfinance as yf
from arch import arch_model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/stock/<ticker>')
def get_stock_data(ticker):
    print(f"Received request for {ticker} at {datetime.now()}")  # Enhanced debug
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    print(f"Fetching data from {start_date} to {end_date}")  # Debug

    try:
        stock = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if stock.empty:
            print(f"No data found for {ticker}")  # Debug
            return jsonify({'error': 'No data found for this ticker'}), 404
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")  # Debug
        return jsonify({'error': str(e)}), 500

    print(f"Columns available: {stock.columns.tolist()}")  # Debug
    if 'Adj Close' in stock.columns:
        price_column = 'Adj Close'
    else:
        price_column = 'Close'

    returns = 100 * stock[price_column].pct_change().dropna()
    print(f"Returns calculated with {len(returns)} points")  # Debug

    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        results = model.fit()
        print(f"GARCH model fitted for {ticker}")  # Debug
    except Exception as e:
        print(f"Error fitting GARCH model for {ticker}: {e}")  # Debug
        return jsonify({'error': f'GARCH error: {str(e)}'}), 500

    dates = returns.index.astype(str).tolist()
    return_data = returns.tolist()
    volatility = results.conditional_volatility.tolist()
    forecast = results.forecast(horizon=5).variance.dropna() ** 0.5
    forecast_dates = pd.date_range(start=dates[-1], periods=6, freq='B')[1:].astype(str).tolist()
    forecast_values = forecast.iloc[-1].tolist()

    return jsonify({
        'dates': dates,
        'returns': return_data,
        'volatility': volatility,
        'forecast_dates': forecast_dates,
        'forecast_values': forecast_values
    })

if __name__ == '__main__':
    print("Starting Flask app...")  # Debug
    app.run(debug=True, host='0.0.0.0', port=5000)