#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 300)


# In[3]:


# Chemin du fichier
chemin = "merged.csv"

# Lecture du fichier CSV
df = pd.read_csv(chemin)

# Affichage des premières lignes
print(df.head())


# In[4]:


import pandas as pd
import ast

# Assuming 'df' is your DataFrame
# Function to safely convert a string representation of a list to an actual list
def convert_to_list(genre):
    if isinstance(genre, str):  # Check if it's a string
        try:
            genre = ast.literal_eval(genre)  # Convert string to list if valid
        except (ValueError, SyntaxError):
            pass  # If it fails, leave the genre as is (it's already a valid string or malformed)
    if not isinstance(genre, list):  # Ensure the value is a list, else convert it to an empty list
        genre = []
    return genre

# Apply the function to the Genre column
df['Genre'] = df['Genre'].apply(convert_to_list)

# Check if all entries are lists
are_all_lists = df['Genre'].apply(lambda x: isinstance(x, list)).all()

# Display the result
print(f"All genres are lists: {are_all_lists}")


# In[5]:


df.head


# In[6]:


# Create a new DataFrame with only Title, Genre, and Summary
new_df = df[["Title", "Genre", "Summary"]]
new_df.head()


# In[7]:


def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text


# In[8]:


new_df['clean_summary'] = new_df['Summary'].apply(lambda x: clean_text(x))
new_df.head()


# In[9]:


def freq_words(x, terms = 30): 
    all_words = ' '.join([text for text in x]) 
    all_words = all_words.split() 
    fdist = nltk.FreqDist(all_words) 
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
    d = words_df.nlargest(columns="count", n = terms) 
    # visualize words and frequencies
    plt.figure(figsize=(12,15)) 
    ax = sns.barplot(data=d, x= "count", y = "word") 
    ax.set(ylabel = 'Word') 
    plt.show()
  
# print 100 most frequent words 
freq_words(new_df['Summary'], 100)


# In[10]:


import pandas as pd
from collections import Counter

# Assuming 'df' is your DataFrame and 'Genre' column contains lists

# Step 1: Flatten the lists in the 'Genre' column into a single list
all_genres = [genre for sublist in new_df['Genre'] for genre in sublist]

# Step 2: Count the frequency of each genre
genre_counts = Counter(all_genres)

# Step 3: Display the most common genres
most_common_genres = genre_counts.most_common()

# Show the results
print("Most frequent genres:")
for genre, count in most_common_genres:
    print(f"{genre}: {count}")


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Assuming 'df' is your DataFrame and 'Genre' column contains lists

# Step 1: Flatten the lists in the 'Genre' column into a single list
all_genres = [genre for sublist in df['Genre'] for genre in sublist]

# Step 2: Count the frequency of each genre
genre_counts = Counter(all_genres)

# Step 3: Get the top 50 most common genres
top_50_genres = genre_counts.most_common(50)

# Step 4: Prepare data for the plot
genres, counts = zip(*top_50_genres)

# Step 5: Create a bar plot for the top 50 genres, reverse the order for most frequent at the top
plt.figure(figsize=(12, 8))
plt.barh(genres[::-1], counts[::-1], color='skyblue')  # Reverse the order here
plt.xlabel('Frequency')
plt.ylabel('Genre')
plt.title('Top 50 Most Frequent Genres')
plt.tight_layout()

# Show the plot
plt.show()


# In[12]:


nltk.download('stopwords')


# In[13]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

new_df['clean_summary'] = new_df['clean_summary'].apply(lambda x: remove_stopwords(x))


# In[14]:


freq_words(new_df['clean_summary'], 100)


# In[15]:


from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(new_df['Genre'])

# transform target variable
y = multilabel_binarizer.transform(new_df['Genre'])


# In[16]:


# used the 10,000 most frequent words in the data as my features. 

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)


# In[17]:


# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(new_df['clean_summary'], y, test_size=0.2, random_state=9)


# In[18]:


# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)


# In[19]:


from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score


# In[20]:


lr = LogisticRegression()
clf = OneVsRestClassifier(lr)


# In[21]:


# fit model on train data
clf.fit(xtrain_tfidf, ytrain)


# In[22]:


# make predictions for validation set
y_pred = clf.predict(xval_tfidf)


# In[23]:


y_pred[3]


# In[24]:


multilabel_binarizer.inverse_transform(y_pred)[3]


# In[25]:


# evaluate performance
f1_score(yval, y_pred, average="micro")


# In[26]:


# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)


# In[27]:


t = 0.3 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)


# In[28]:


f1_score(yval, y_pred_new, average="micro")


# In[29]:


def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)


# In[30]:


for i in range(5): 
    k = xval.sample(1).index[0] 
    print("Book: ", new_df['Title'][k], "\nPredicted genre: ", infer_tags(xval[k])), print("Actual genre: ",new_df['Genre'][k], "\n")

