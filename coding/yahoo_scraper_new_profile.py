# filename: yahoo_scraper_new_profile.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Set up the WebDriver
driver_path = '/home/lxh/dev/withautogen/chromedriver-linux/chromedriver'
service = Service(executable_path=driver_path)

# Create ChromeOptions with new profile
options = Options()
options.add_argument("user-data-dir=/home/lxh/dev") 

driver = webdriver.Chrome(service=service, options=options)

# Navigate to Yahoo.com
driver.get("https://www.yahoo.com/")

# Find the search bar element
search_bar = driver.find_element(By.ID, "ybar-sbq")

# Extract the placeholder text
placeholder_text = search_bar.get_attribute("placeholder")

# Print the extracted text
print(f"Placeholder Text: {placeholder_text}")

# Close the browser
driver.quit()