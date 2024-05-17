# filename: sohu_news.py
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Set up Chrome options for headless mode
options = Options()
options.add_argument('--headless')
#options.add_argument('--disable-gpu')
#options.add_argument('--remote-debugging-pipe')
options.add_argument('--no-sandbox')  # Ensure Chrome runs without sandboxing
options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems 

# Specify the paths to Chrome and Chromedriver
chrome_path = '/home/lxh/dev/withautogen/chrome-linux64/chrome'
chromedriver_path = '/home/lxh/dev/withautogen/chromedriver-linux64/chromedriver'

options.binary_location = chrome_path
service = Service(executable_path=chromedriver_path)
# Create a WebDriver instance
driver = webdriver.Chrome(service=service, options=options)

# Navigate to Sohu.com
driver.get("https://www.sohu.com/")

# Find news elements and extract titles and links
news_elements = driver.find_elements(By.CSS_SELECTOR, "a")
news_list = []
for element in news_elements:
    news_item = {
        'title': element.text,
        'link': element.get_attribute('href')
    }
    news_list.append(news_item)

# Print the collected news
for news in news_list:
    print(f"Title: {news['title']}")
    print(f"Link: {news['link']}")
    print("-----")

# Close the WebDriver
driver.quit()