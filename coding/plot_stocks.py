# filename: plot_stocks.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    stock = yf.Ticker(symbol)
    history = stock.history(start=start_date, end=end_date)
    return history

def main():
    start_date = "2022-01-01"
    end_date = "2024-03-31"  # Adjust this in the future as needed
    
    # Fetch historical data
    nvda_data = fetch_stock_data("NVDA", start_date, end_date)
    tsla_data = fetch_stock_data("TSLA", start_date, end_date)
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(nvda_data.index, nvda_data['Close'], label='NVDA (NVIDIA)')
    plt.plot(tsla_data.index, tsla_data['Close'], label='TSLA (Tesla)')
    
    plt.title('Stock Prices of NVDA and TSLA from Jan 2022 to Mar 2024')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()