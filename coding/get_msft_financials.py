# filename: get_msft_financials.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Set up Chrome options for headless mode
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')  # Add this argument
options.binary_location = '/snap/bin/chromium'

# Specify the path to ChromeDriver
chromedriver_path = "/usr/bin/chromedriver"  

# Initialize the webdriver using the specified ChromeDriver path
driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)

# Navigate to Yahoo Finance page for Microsoft
driver.get('https://finance.yahoo.com/quote/MSFT/')

# Locate and extract financial data elements (example: market cap)
market_cap_element = driver.find_element(By.XPATH, '//*[@id="quote-summary"]/div[1]/table/tbody/tr[1]/td[2]')
market_cap = market_cap_element.text

# Print the extracted data
print("Market Cap:", market_cap)

# Close the browser
driver.quit()