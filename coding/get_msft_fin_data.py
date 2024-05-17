# filename: get_msft_fin_data.py
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up Chrome options for headless mode
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')

# Add user-data-dir to address potential profile-related issues
user_data_dir = '/tmp/chrome_user_data'  
options.add_argument(f'--user-data-dir={user_data_dir}')

# Create the user data directory if it doesn't exist
if not os.path.exists(user_data_dir):
    os.makedirs(user_data_dir)

# Specify the paths to Chrome and Chromedriver (manually, as automatic detection seems to be failing)
chrome_path = '/home/lxh/dev/withautogen/chrome-linux64/chrome'
chromedriver_path = '/home/lxh/dev/withautogen/chromedriver-linux64/chromedriver'  # Ensure this path is correct

options.binary_location = chrome_path

# Use the specified Chromedriver path
driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)

# Navigate to Yahoo Finance page for Microsoft
driver.get('https://finance.yahoo.com/quote/MSFT/')

# Wait for the element to be present and clickable, using a more specific and potentially more stable selector 
try:
    wait = WebDriverWait(driver, 10)  # Adjust wait time if needed
    price_element = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'fin-streamer[data-symbol="MSFT"] .Fw\(b\).Fz\(36px\)')))  
    current_price = price_element.text
    print("Current Price:", current_price)

    # Extract other financial data using similar logic and appropriate CSS selectors
    # ...

except Exception as e:
    print("Error:", e)

# Print browser logs for debugging (if needed)
for entry in driver.get_log('browser'):
    print(entry)

# Close the browser
driver.quit()