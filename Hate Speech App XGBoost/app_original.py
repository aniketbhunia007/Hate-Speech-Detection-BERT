from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

import xgboost

import pandas as pd
import numpy as np
import pickle
import sys
import nltk

from nltk.stem.porter import *
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
import logging
from gensim.models import Word2Vec



def clean_remove_b(data):  
  data.rename(columns={'text':'tweet'},inplace=True)
  data['tweet'] = data['tweet'].astype(str)

  data['tweet'] = data['tweet'].apply(lambda x:x[2:] if x[0:2]=="b'" or 'b"' else x)

def preprocess(data):
  data['tweet'] = data['tweet'].apply(lambda x:' '.join(i for i in [a for a in x.split() if a.find('@')==-1]))
  data['tweet'] = data['tweet'].apply(lambda x:' '.join(i for i in [a for a in x.split() if a.find('http')==-1]))
  
  ## we are removing hashtags now, but while doing transfer learning, to learn the embeddings we didnt remove these, 
  ## just to include such words in our vocabulary
  
  data['tweet'] = data['tweet'].apply(lambda x:' '.join(i for i in [a for a in x.split() if a.find('#')==-1]))
  data['tweet'] = data['tweet'].apply(lambda x:''.join([i for i in x if not i.isdigit()]))
  data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
  data['tweet'] = data['tweet'].str.replace('[^\w\s]','')

  
  stop = stopwords.words('english')
  data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

  remove_word = ['rt','mkr','im']
  data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in remove_word))

def preprocess_2(data):
  data['tweet'] = data['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
  data['tweet'].head()

## I guess we shall write a function for embedding conversion. but would it work just like that, I should it should.
# i mean we can keep the embeddings loaded outside the app function, so that every time a new query comes up, it shouldnt load it again
wv = gensim.models.KeyedVectors.load_word2vec_format("./model_transfer_learning.txt", binary=False)
wv.init_sims(replace=True)

def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens
    



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    # train = pd.read_csv("train.csv")
    # preprocess(train)
    # preprocess_2(train)

    # tfidf = TfidfVectorizer(max_features=100, lowercase=True, analyzer='word',
    # stop_words= 'english',ngram_range=(1,1))
    # print(train['tweet'])
    # tfidf.fit(train['tweet'])
# soemthing wrong here

# word embeddings XGBoost model
    model = open('xgb_final_model.pkl','rb')
    clf = joblib.load(model)

# TF-IDF model
    # model = open('xgb_model_tfidf.pkl','rb')
    # clf = joblib.load(model)
    
    if request.method == 'POST':
        message = request.form['message']
        data = pd.DataFrame([str(message)],columns=['tweet'])

        print(data)
        
        preprocess(data)
        preprocess_2(data)
        
        print(data)
        vect = data.apply(lambda r: w2v_tokenize_text(r['tweet']), axis=1).values
        vect = word_averaging_list(wv,vect)

        print(vect)
        
        my_prediction = clf.predict(vect)
        probab = clf.predict_proba(vect)[::,-1]

        print(clf.predict_proba(vect),'hate probability')
        
        return render_template('result.html',prediction = my_prediction, probability = probab*100.00)


if __name__ == '__main__':
	app.run(debug=True)

# as of now just for sake of not changing the whole preprocessing code again we are changing the text input into DataFrame
        # But will change the preprocessing code later
		####
            # code for preprocessing and embeddings of test message
        # clean_remove_b(data)