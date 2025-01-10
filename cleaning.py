#!/usr/bin/env python
# coding: utf-8

# In[6]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime

# Function to clean a dataset based on a specific column value
def clean_dataset(df, column, value_to_remove, output_file):
    """
    Cleans the dataset by removing rows with a specific value in a given column
    and performs additional cleaning operations.
    """
    # Display counts before cleaning
    counts = df[column].value_counts()
    print(f"Counts of {column} values before cleaning:")
    print(counts)
    
    # Remove rows with the specified value
    df = df[df[column] != value_to_remove]
    
    # Display counts after cleaning
    counts = df[column].value_counts()
    print(f"Counts of {column} values after cleaning:")
    print(counts)
    
    # Drop null rows
    num_null_rows = df.isna().any(axis=1).sum()
    if num_null_rows != 0:
        df = df.dropna()
        
    # Save the merged dataset into a new CSV file
    df.to_csv(output_file, index=False)
             
    return df

# Example usage
df_not_converted = pd.read_csv('cleaned_books_that_should_be_made_into_movies_details.csv')
df_converted = pd.read_csv('cleaned_books_made_into_movies_details.csv')

df_converted = clean_dataset(df_converted, 'Converted', 0,'cleaned_books_made_into_movies_details.csv' )
print(df_converted.head())
df_not_converted = clean_dataset(df_not_converted, 'Converted', 1, 'cleaned_books_that_should_be_made_into_movies_details.csv')
print(df_not_converted.head())

def merge_datasets(file_converted, file_not_converted, output_file="merged.csv"):
    # Load the two datasets
    df_converted = pd.read_csv(file_converted)
    df_not_converted = pd.read_csv(file_not_converted)
    
    # Merge the datasets
    df = pd.concat([df_converted, df_not_converted], ignore_index=True)
    
    # Lowercase and clean the 'Summary' column using .loc[]
    if 'Summary' in df.columns:
        df.loc[:, 'Summary'] = df['Summary'].str.lower().str.replace('[^\w\s]', '', regex=True)
    
    # Extract numeric pages
    def extract_pages(pages_str):
        try:
            return int(pages_str.split()[0])
        except (ValueError, IndexError):
            return None
    
    if 'Pages' in df.columns:
        df.loc[:, 'num_pages'] = df['Pages'].apply(extract_pages)
        df = df.dropna(subset=['num_pages'])
    
    # Clean and convert publication date
    def clean_and_convert_date(date_str):
        cleaned_str = date_str.replace("Published", "").strip()
        try:
            return pd.to_datetime(cleaned_str, errors='raise')
        except (ValueError, pd.errors.OutOfBoundsDatetime):
            try:
                return datetime.strptime(cleaned_str, '%B %d, %Y')
            except ValueError:
                return pd.NaT
    
    if 'Publication Date' in df.columns:
        df['Publication Date'] = df['Publication Date'].apply(clean_and_convert_date)
        df = df.dropna(subset=['Publication Date'])

    # Save the merged dataset into a new CSV file
    df.to_csv(output_file, index=False)
    
    # Return the first 5 records for a quick view
    return df.head()

# Call the function
merge_datasets('cleaned_books_made_into_movies_details.csv', 'cleaned_books_that_should_be_made_into_movies_details.csv')

# Load the merged CSV file and print the first 5 records
merged = pd.read_csv('merged.csv')
print(merged.head())


# In[ ]:





# In[ ]:




