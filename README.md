# Book to Movie Conversion & Genre Prediction

## Objective

The main goal of this project is to explore how the information in book summaries can be used to predict whether a book will be adapted into a movie and to predict the genre of the book itself. It focuses on the analysis and prediction of book-to-movie adaptations, using Natural Language Processing (NLP) techniques on book summaries. We collected and cleaned a dataset of books, verifying whether each one was adapted into a movie. Based on this data, two machine learning models were created: **Conversion Model**,**Genre Prediction Model**

## Project Structure:

- **scraping.py:** Scraping and gathering data on books and their movie adaptations.
- **book_to_movie_checker.py:** Adding the "converted" column that indicates whether a book has a movie adaptation.
- **cleaning.py:** Cleaning and preprocessing the data.
- **conversion_model.py:** This model predicts whether a book has been adapted into a movie based on its summary. The model was trained to identify features in the text that suggest the likelihood of a film adaptation.
- **genre_prediction_model.py:** The second model predicts the genre of the book, also based on its summary. This model helps categorize books based on the themes discussed in their text.

## Dataset

The project includes three main CSV files containing the dataset:

1. **cleaned_books_made_into_movies_details.csv**  
   This file contains details of books that have been adapted into movies. It includes information about the book and its corresponding movie adaptation.

2. **cleaned_books_that_should_be_made_into_movies_details.csv**  
   This file contains details of books that have not yet been adapted into movies. These books are potential candidates for film adaptations.

3. **merged.csv**  
   A cleaned version that combines both the books that have been adapted into movies and those that have not. This file serves as the primary dataset used for training and analysis.
   
## Libraries Used:

- Pandas
- Numpy
- Scikit-learn
- datetime
- nltk
- re
- tqdm
- transformers
- datasets
- Natural Language Processing (NLP) for analyzing book summaries
- Matplotlib and Seaborn 



