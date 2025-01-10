#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# En-tête pour les requêtes HTTP
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

# Fonction pour extraire les détails d'un livre
def get_book_details(book_url):
    try:
        time.sleep(2)  # Délai pour éviter d'être bloqué
        book_response = requests.get(book_url, headers=headers)
        if book_response.status_code == 200:
            book_soup = BeautifulSoup(book_response.content, 'html.parser')

            # Titre
            title = book_soup.find('h1', class_="Text Text__title1")
            title = title.text.strip() if title else "Titre indisponible"

            # Auteur
            author = book_soup.find('span', class_='ContributorLink__name')
            author = author.text.strip() if author else "Auteur indisponible"

            # Date de publication
            pub_date = book_soup.find('p', {'data-testid': 'publicationInfo'})
            pub_date = pub_date.text.replace('First published ', '').strip() if pub_date else "Date de publication indisponible"

            # Nombre de pages
            num_pages = book_soup.find('p', {'data-testid': 'pagesFormat'})
            num_pages = num_pages.text.split(',')[0].strip() if num_pages else "Non renseigné"

            # Genres - Recherche dans la liste déroulante des genres
            genre_container = movie_soup.find('div', {'data-testid': 'genresList'})
            genres = [g.text.strip() for g in genre_container.find_all('a')] if genre_container else []

            # Limiter à 3 genres seulement
            genres = genres[:3] if genres else ["Genre indisponible"] * 3
            
            # Note
            rating = book_soup.find('div', class_="RatingStatistics__rating")
            rating = rating.text.strip() if rating else "Note indisponible"

            # Résumé
            summary_tag = book_soup.find('div', class_="DetailsLayoutRightParagraph__widthConstrained")
            summary = summary_tag.text.strip() if summary_tag else "Résumé indisponible"

            return {
                'Title': title,
                'Author': author,
                'Publication Date': pub_date,
                'Pages': num_pages,
                'Genre': genre,
                'Rating': rating,
                'Summary': summary
            }
        else:
            logging.warning(f"Impossible d'accéder à la page du livre : {book_url} (Code {book_response.status_code})")
    except Exception as e:
        logging.error(f"Erreur lors de la récupération de la page du livre : {book_url}\n{e}")
    return None

# Fonction pour scraper une page spécifique
def scrape_page(base_url, page_number):
    url = f"{base_url}?page={page_number}"
    try:
        time.sleep(2)  # Délai pour éviter d'être bloqué
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            books = soup.select('a.bookTitle')
            authors = soup.select('a.authorName')

            books_data = []
            for book, author in zip(books, authors):
                book_url = "https://www.goodreads.com" + book['href']
                logging.info(f"Scraping book details from {book_url}")
                book_details = get_book_details(book_url)
                if book_details:
                    books_data.append(book_details)

            return books_data
        else:
            logging.warning(f"Erreur lors de la récupération de la page {page_number} : {response.status_code}")
    except Exception as e:
        logging.error(f"Erreur lors de la récupération de la page {page_number} : {e}")
    return []

# Fonction pour scraper les livres pour une catégorie
def scrape_books(category_name, base_url, max_pages=10):
    all_books_data = []
    for page in range(1, max_pages + 1):
        logging.info(f"Scraping page {page} pour {category_name}...")
        books_data = scrape_page(base_url, page)
        all_books_data.extend(books_data)

    # Convertir en DataFrame
    df_books = pd.DataFrame(all_books_data)

    # Nettoyer les données
    if not df_books.empty:
        columns_to_check = ['Title', 'Author', 'Publication Date', 'Pages', 'Genre', 'Rating', 'Summary']
        for col in columns_to_check:
            df_books = df_books[~df_books[col].isin(["indisponible", "Non renseigné"])]
        df_books = df_books.dropna(subset=columns_to_check)

        # Sauvegarder les résultats
        output_file = f"cleaned_{category_name.replace(' ', '_').lower()}_details.csv"
        df_books.to_csv(output_file, index=False)
        logging.info(f"Les données nettoyées pour {category_name} ont été sauvegardées dans {output_file}")
    else:
        logging.warning(f"Aucune donnée valide n'a été récupérée pour {category_name}.")

# Scraper les deux catégories
scrape_books("Books Made Into Movies", "https://www.goodreads.com/list/show/252.Books_Made_into_Movies", max_pages=10)
scrape_books("Books That Should Be Made Into Movies", "https://www.goodreads.com/list/show/1043.Books_That_Should_Be_Made_Into_Movies", max_pages=10)

