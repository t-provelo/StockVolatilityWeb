from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_caching import Cache
import yfinance as yf
from arch import arch_model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 300})  # Cache for 5 minutes

@app.route('/stock/<ticker>')
@cache.cached(timeout=300)  # Cache each ticker's data for 5 minutes
def get_stock_data(ticker):
    print(f"Received request for {ticker} at {datetime.now()}")
    days = int(request.args.get('days', 365))  # Default to 365 days, adjustable via query param
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    print(f"Fetching data from {start_date} to {end_date} for {days} days")

    try:
        stock = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if stock.empty:
            print(f"No data found for {ticker}")
            return jsonify({'error': f'No data found for {ticker}'}), 404
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return jsonify({'error': f'Failed to fetch data: {str(e)}'}), 500

    print(f"Columns available: {stock.columns.tolist()}")
    if 'Adj Close' in stock.columns:
        price_column = 'Adj Close'
    else:
        price_column = 'Close'

    returns = 100 * stock[price_column].pct_change().dropna()
    print(f"Returns calculated with {len(returns)} points")

    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        results = model.fit()
        print(f"GARCH model fitted for {ticker}")
    except Exception as e:
        print(f"Error fitting GARCH model for {ticker}: {e}")
        return jsonify({'error': f'GARCH error: {str(e)}'}), 500

    dates = returns.index.astype(str).tolist()
    return_data = returns.to_numpy().tolist()
    print(f"Returns data: {return_data[:5]}...")  # Debug
    volatility = results.conditional_volatility.to_numpy().tolist()
    forecast = results.forecast(horizon=5).variance.dropna() ** 0.5
    forecast_dates = pd.date_range(start=dates[-1], periods=6, freq='B')[1:].astype(str).tolist()
    forecast_values = forecast.iloc[-1].to_numpy().tolist()

    return jsonify({
        'dates': dates,
        'returns': return_data,
        'volatility': volatility,
        'forecast_dates': forecast_dates,
        'forecast_values': forecast_values
    })

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)