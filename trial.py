import json

# Open the JSON file and load its contents into a Python object
with open('articles4.json', 'r') as f:
    data = json.load(f)
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
import warnings
warnings.filterwarnings('ignore')

#list of all years and link of all the years for bombay high court since 1800 to 2023.
year_list=[]
url='https://indiankanoon.org/browse/bombay/'
response = requests.get(url)
html=response.text
soup = BeautifulSoup(html, 'html.parser')
link=soup.find_all('div', class_='browselist')
# print((link[0].a).text)
# print('https://indiankanoon.org'+(link[0].a).get('href'))
for i in range(len(link)):
    year_list.append((link[i].a).text)
print(year_list)
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# define stop words
stop_words = set(stopwords.words('english'))

# function to process text
def process_text(text):
    if text!=None:
        words = text.lower().split()
        return [word for word in words if word not in stop_words]
    
# define lists for titles, links, and texts
titles = data['article_name']
links = data['link']
texts = data['text']
num_articles= 10
year= year_list

#for one single list
# preprocess the texts using the process_text function
processed_texts = []
for text in texts:
    if text is not None:
        processed_text = process_text(text)
        if processed_text:
            processed_texts.append(processed_text)

#print(processed_texts[0])
import re

# define regular expression pattern to match non-word characters
pattern = re.compile(r'[^\w\s]')

# remove non-word characters from processed texts
#processed_texts = [[word for word in text if not pattern.match(word)] for text in processed_texts]
processed_texts = [[word for word in text if isinstance(word, str) and not pattern.match(word)] for text in processed_texts]

# convert the processed texts into a document-term matrix using TF-IDF vectorization
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform([' '.join(text) for text in processed_texts])
# Filter out empty lists from processed_texts
processed_texts_filtered = [article for article in processed_texts if article]

#print(x)

# calculate cosine similarities between the article of interest and all other articles
article_text = 'https://indiankanoon.org/docfragment/360021/'
article_index = links.index(article_text)
cosine_similarities = cosine_similarity(x[article_index], x).flatten()

similarities_df = pd.DataFrame(columns=['title', 'link', 'text', 'similarity'])

for idx, cosine_sim in enumerate(cosine_similarities):
    similarities_df = pd.concat([similarities_df, pd.DataFrame({'title': titles[idx], 'link': links[idx], 'text': texts[idx], 'similarity': cosine_sim}, index=[idx])])

similarities_df = similarities_df.sort_values(by='similarity', ascending=False)
similarities_df = similarities_df[similarities_df.text != article_text]
#print links and names of similar articles
print(f'Top {num_articles} similar articles to "{article_text}" published in {year}:')
if len(similarities_df) == 0:
    print('No similar articles found in this year')
else:
    for index, row in similarities_df.iterrows():
        print(row['title'])
        print(row['link'])
        print('Similarity score:', round(row['similarity'], 3))
        print('\n')
print(similarities_df.head(10))

import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# load the Word2Vec model
model = Word2Vec.lomodel = Word2Vec(processed_texts, window=5, min_count=1, workers=4)

# define stop words
stop_words = set(stopwords.words('english'))

# function to process text
def process_text(text):
    if text!=None:
        words = text.lower().split()
        return [word for word in words if word not in stop_words]

# define empty lists for titles, links, and texts
titles = data['article_name']
links = data['link']
texts = data['text']
num_articles= 10
year= year_list

# preprocess the texts using the process_text function
processed_texts = []
for text in texts:
    if text is not None:
        processed_text = process_text(text)
        if processed_text:
            processed_texts.append(processed_text)

# remove non-word characters from processed texts
pattern = re.compile(r'[^\w\s]')
processed_texts = [[word for word in text if not pattern.match(word)] for text in processed_texts]

# convert the processed texts into a document-term matrix using Word2Vec
document_vectors = []
for text in processed_texts:
    vector = []
    for word in text:
        if word in model.wv.key_to_index:
            vector.append(model.wv[word])
    vector = np.mean(vector, axis=0)
    document_vectors.append(vector)

# calculate cosine similarities between the article of interest and all other articles
article_text = 'https://indiankanoon.org/docfragment/61902/'
article_index = links.index(article_text)
cosine_similarities1 = cosine_similarity([document_vectors[article_index]], document_vectors).flatten()

similarities_df1 = pd.DataFrame(columns=['title', 'link', 'text', 'similarity'])

for idx, cosine_sim in enumerate(cosine_similarities1):
    similarities_df1 = pd.concat([similarities_df1, pd.DataFrame({'title': titles[idx], 'link': links[idx], 'text': texts[idx], 'similarity': cosine_sim}, index=[idx])])

similarities_df1 = similarities_df1.sort_values(by='similarity', ascending=False)
similarities_df1 = similarities_df1[similarities_df1.text != article_text]
#print links and names of similar articles
print(f'Top {num_articles} similar articles to "{article_text}" published in {year}:')
if len(similarities_df1) == 0:
    print('No similar articles found in this year')
else:
    for index, row in similarities_df1.iterrows():
        print(row['title'])
        print(row['link'])
        print('Similarity score:', round(row['similarity'], 3))
        print('\n')
print(similarities_df1.head(10))
