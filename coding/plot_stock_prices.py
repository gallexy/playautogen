# filename: plot_stock_prices.py
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
nvda_data = pd.read_csv('NVDA_data.csv', index_col='Date', parse_dates=True)
tesla_data = pd.read_csv('TSLA_data.csv', index_col='Date', parse_dates=True)

# Plot the 'Close' price for both stocks
plt.figure(figsize=(10, 6))
plt.plot(nvda_data['Close'], label='NVDA')
plt.plot(tesla_data['Close'], label='TSLA')
plt.title('NVDA and TSLA Stock Prices (Jan 2022 - Nov 2023)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()