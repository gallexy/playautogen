# filename: sohu_news_scraper.py

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

# Set up Chrome options
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Set up Chrome driver
driver = webdriver.Chrome(options=options, executable_path=ChromeDriverManager().install())

# Navigate to Sohu news page
driver.get('https://news.sohu.com/')

# Wait for the page to load
time.sleep(5)

# Find all news titles on the page
news_titles = driver.find_elements(By.XPATH, '//h4/a')

# Print the news titles
for title in news_titles:
    print(title.text)

# Close the driver
driver.quit()

print("TERMINATE")