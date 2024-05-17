# filename: tesla_finance.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager  # Import WebDriver manager
from selenium.webdriver.common.by import By
import time

# Set up Chrome WebDriver in headless mode
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox') 

# Set the path to the Chromium binary
options.binary_location = '/snap/bin/chromium' 

# Use WebDriver manager to get the appropriate chromedriver
service = Service(ChromeDriverManager().install()) 
driver = webdriver.Chrome(service=service, options=options)

# The rest of the code remains the same as before...