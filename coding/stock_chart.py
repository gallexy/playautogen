# filename: stock_chart.py

import yfinance as yf
import matplotlib.pyplot as plt

# Define the ticker symbols
tickers = ["NVDA", "TSLA"]

# Download historical data for the past 2.5 years
start_date = "2022-01-01"
end_date = "2024-03-31"
data = yf.download(tickers, start=start_date, end=end_date)

# Calculate the daily percentage change for each ticker
def pct_change(x):
    try:
        return x['Adj Close'].pct_change()
    except KeyError:
        return x.pct_change()

data = data.apply(pct_change)

# Plot the chart
fig, ax = plt.subplots()
for ticker in tickers:
    data[ticker].plot(ax=ax, label=ticker)

# Add legend and grid
ax.legend()
ax.grid(True)

# Show the plot
plt.show()