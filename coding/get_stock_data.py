# filename: get_stock_data.py
import yfinance as yf

# Define the start and end dates
start_date = '2022-01-01'
end_date = '2023-11-30'  # Updated end date due to data limitations

# Download the data
nvda_data = yf.download('NVDA', start=start_date, end=end_date)
tesla_data = yf.download('TSLA', start=start_date, end=end_date)

# Save the data to CSV files
nvda_data.to_csv('NVDA_data.csv')
tesla_data.to_csv('TSLA_data.csv')

print("Stock data downloaded successfully!")