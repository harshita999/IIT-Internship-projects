from flask import Flask, render_template, request
import json
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

# define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        year = request.form['year']
        # your search function here
        results = {}  # replace with your search function that returns results
        return render_template('search.html', results=results)
    else:
        return render_template('search.html')

# run the app
if __name__ == '__main__':
    app.run(debug=True)
