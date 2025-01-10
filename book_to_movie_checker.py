#!/usr/bin/env python
# coding: utf-8

# In[2]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time

def check_converted(csv_file_path):
    # Load the dataset
    df = pd.read_csv(csv_file_path)
    df['Converted'] = 0  # Initialize the 'Converted' column to 0
    df = df[df['Title'] != "Золотой теленок"]
    
    # Chrome driver settings
    chrome_options = Options()
    chrome_options.add_experimental_option('prefs', {'intl.accept_languages': 'en'})
    chrome_options.add_argument("--disable-search-engine-choice-screen")
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        url = "https://www.allocine.fr/"
        driver.get(url)
        
        # Close popups
        def close_popups():
            try:
                cookies_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="cmp-main"]/button[2]'))
                )
                cookies_button.click()
            except Exception as e:
                print(f"No pop-up or unable to close: {e}")
        
        close_popups()
        
        # Iterate over each row in the DataFrame
        for i, row in df.iterrows():
            try:
                search_button = driver.find_element(By.XPATH, '/html/body/div[2]/header/div/div[1]/div[4]')
                driver.execute_script("arguments[0].click();", search_button)
                
                search_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.NAME, "q"))
                )
                
                title = row['Title']
                search_input.clear()
                search_input.send_keys(title)
                search_input.send_keys(Keys.ENTER)
                
                time.sleep(2)
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Check for "Oups!" error message
                oups_element = soup.find('div', class_='content-txt wrong-search-msg')
                if oups_element and "Oups" in oups_element.text:
                    df.at[i, 'Converted'] = 0
                else:
                    df.at[i, 'Converted'] = 1
                
                print(f"Processed: {title}, Converted: {df.at[i, 'Converted']}")
            
            except Exception as e:
                print(f"Error in searching or processing the title {title}: {e}")
        
    finally:
        # Save the updated DataFrame
        df.to_csv(csv_file_path, index=False)
        driver.quit()
        print("Process completed and CSV updated.")


check_converted('cleaned_books_made_into_movies_details.csv')
df=pd.read_csv('cleaned_books_made_into_movies_details.csv')
counts = df['Converted'].value_counts()
print("Counts of Converted values:")
print(counts)

def check_not_converted(csv_file_path):
    # Load the dataset
    df = pd.read_csv(csv_file_path)
    df['Converted'] = 0  # Initialize the 'Converted' column to 0
    
    # Chrome driver settings
    chrome_options = Options()
    chrome_options.add_experimental_option('prefs', {'intl.accept_languages': 'en'})
    chrome_options.add_argument("--disable-search-engine-choice-screen")
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        url = "https://www.allocine.fr/"
        driver.get(url)
        
        # Close popups
        def close_popups():
            try:
                cookies_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="cmp-main"]/button[2]'))
                )
                cookies_button.click()
            except Exception as e:
                print(f"No pop-up or unable to close: {e}")
        
        close_popups()
        
        # Iterate over each row in the DataFrame
        for i, row in df.iterrows():
            try:
                search_button = driver.find_element(By.XPATH, '/html/body/div[2]/header/div/div[1]/div[4]')
                driver.execute_script("arguments[0].click();", search_button)
                
                search_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.NAME, "q"))
                )
                
                title = row['Title']
                search_input.clear()
                search_input.send_keys(title)
                search_input.send_keys(Keys.ENTER)
                
                time.sleep(2)
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Check for "Oups!" error message
                oups_element = soup.find('div', class_='content-txt wrong-search-msg')
                if oups_element and "Oups" in oups_element.text:
                    df.at[i, 'Converted'] = 0
                else:
                    result_titles = soup.find_all('a', class_='meta-title-link')
                    exact_match_found = any(title.lower() == result.get_text(strip=True).lower() for result in result_titles)
                    
                    if exact_match_found:
                        df.at[i, 'Converted'] = 1
                    else:
                        df.at[i, 'Converted'] = 0
                
                print(f"Processed: {title}, Converted: {df.at[i, 'Converted']}")
            
            except Exception as e:
                print(f"Error in searching or processing the title {title}: {e}")
        
    finally:
        # Save the updated DataFrame
        df.to_csv(csv_file_path, index=False)
        driver.quit()
        print("Process completed and CSV updated.")

# Example usage
check_not_converted('cleaned_books_that_should_be_made_into_movies_details.csv')
df=pd.read_csv('cleaned_books_that_should_be_made_into_movies_details.csv')
counts = df['Converted'].value_counts()
print("Counts of Converted values:")
print(counts)

