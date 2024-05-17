# filename: get_msft_finance.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# Set up Chrome options for headless mode
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument("--window-size=1920,1200")
options.add_argument('--no-sandbox')  # Add this line
options.add_argument('--disable-dev-shm-usage')  # Add this line
# Add the path to Chromium
options.binary_location = "/snap/bin/chromium"

# Set the path to chromedriver 
service = Service(executable_path='/usr/bin/chromedriver') 

# Create a WebDriver instance
driver = webdriver.Chrome(service=service, options=options)

# ... (rest of the code remains the same) ... 