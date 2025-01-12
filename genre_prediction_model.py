#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import re
import ast
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV


# Loading the data

chemin = "merged.csv"
df = pd.read_csv(chemin)

# Preprocessing functions
def convert_to_list(genre):
    if isinstance(genre, str):
        try:
            genre = ast.literal_eval(genre)
        except (ValueError, SyntaxError):
            pass
    if not isinstance(genre, list):
        genre = []
    return genre

# Apply genre conversion to list
df['Genre'] = df['Genre'].apply(convert_to_list)

# Remove 'Fiction' genre
df['Genre'] = df['Genre'].apply(lambda genres: [genre for genre in genres if 'Fiction' not in genre])

# Text cleaning
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)  # remove non-alphabetic characters
    text = text.lower()  # lowercasing
    return ' '.join(text.split())

df['clean_summary'] = df['Summary'].apply(lambda x: clean_text(x))

# Remove stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

df['clean_summary'] = df['clean_summary'].apply(lambda x: remove_stopwords(x))


# MultiLabel Binarizer
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df['Genre'])
y = multilabel_binarizer.transform(df['Genre'])

# Splitting dataset into train and validation
xtrain, xval, ytrain, yval = train_test_split(df['clean_summary'], y, test_size=0.2, random_state=9)

# Parameter grid for tuning
param_grid = {
    'vectorizer__max_features': [2000, 5000, 8000],  # TF-IDF features
    'vectorizer__max_df': [0.75, 0.8, 0.85],
    'vectorizer__min_df': [0.01, 0.05],
    'classifier__estimator__C': [0.1, 1, 10],  # Logistic Regression regularization
    'classifier__estimator__max_iter': [100, 300, 500],  # Logistic Regression iterations
}

# Setup TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, max_features=5000)
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

# Setup Logistic Regression classifier with class weights
lr = LogisticRegression(class_weight='balanced', max_iter=500)

# Create a pipeline with the vectorizer and classifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

pipeline = Pipeline([
    ('vectorizer', tfidf_vectorizer),
    ('classifier', OneVsRestClassifier(lr))
])

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_micro', verbose=2, n_jobs=-1)

# Fit the grid search
grid_search.fit(xtrain, ytrain)

# Best parameters from grid search
print("Best parameters found:", grid_search.best_params_)

# Predict and evaluate the best model on the validation set
y_pred = grid_search.best_estimator_.predict(xval)
f1_micro = f1_score(yval, y_pred, average='micro')
print(f"F1 Score (Micro) on Validation Set: {f1_micro}")

# Adjust threshold and re-evaluate
y_pred_prob = grid_search.best_estimator_.predict_proba(xval)
threshold = 0.4  # Threshold for classification
y_pred_threshold = (y_pred_prob >= threshold).astype(int)

f1_micro_threshold = f1_score(yval, y_pred_threshold, average="micro")
print(f"F1 Score (with threshold) on Validation Set: {f1_micro_threshold}")

# Example of predicting new genres
def infer_tags(q, model, vectorizer, threshold=0.3):
    q_vec = vectorizer.transform([q])
    q_pred_prob = model.predict_proba(q_vec)
    return multilabel_binarizer.inverse_transform((q_pred_prob >= threshold).astype(int))

# Test on some examples from the validation set
for i in range(5):
    k = xval.sample(1).index[0]
    print(f"Book: {df['Title'][k]}")
    predicted_genres = infer_tags(xval[k], grid_search.best_estimator_.named_steps['classifier'], grid_search.best_estimator_.named_steps['vectorizer'])
    print(f"Predicted genres: {predicted_genres}")
    print(f"Actual genres: {df['Genre'][k]}\n")


# In[ ]:





# In[ ]:




