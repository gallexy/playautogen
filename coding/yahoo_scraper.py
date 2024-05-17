# filename: yahoo_scraper.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

def main():
    # Setup ChromeDriver path
    chrome_driver_path = '/home/lxh/dev/withautogen/chromedriver-linux/chromedriver'
    
    # Initialize WebDriver with updated syntax
    service = Service(executable_path=chrome_driver_path)
    driver = webdriver.Chrome(service=service)
    
    # Open Yahoo Website
    driver.get("https://www.yahoo.com")
    
    # Waiting a bit for the page to load
    time.sleep(5)  # Be respectful by not hammering the site; adjust as necessary
    
    # Grab Data (example: headlines)
    headlines = driver.find_elements(By.TAG_NAME, "h3")
    for headline in headlines[:10]:  # Limiting to the first 10 headlines
        print(headline.text)
    
    # Close Browser
    driver.quit()

if __name__ == "__main__":
    main()