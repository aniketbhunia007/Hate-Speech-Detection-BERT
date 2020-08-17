## Hate Speech Detection
* American Express Ignite Project 2019
* Pranav D. Pawar ; Mentor : Lokesh Kumar Kriplani

Detailed documentation and experiments details - [here](./hate_sppech_doc.pdf)

### Flask Web App
* Primary features of API - 
  * Custom Text Input testing - Given a text input, we can generate the probability of hate speech with an F1-Score of 94% (using BERT model)
  * Hashtag analysis - 
    * Given a valid hashtag, API scrapes the latest n tweets for that hashtag and performs an evaluation on it using our deployed model. 
    * Finally generates a sorted list of tweets according to their hate probability.
    * Here the input is a hashtag, no. of tweets, and date since you want to perform the evaluation upon
  * User analysis
    * Given a valid twitter user ID, API scrapes the latest n tweets on the user’s timeline and similar to the previous case generates a table of a sorted list of tweets according to their hate probability.
    * Here the input is only the hashtag and no. of tweets to scrape

* BERT App Service
  * `cd Hate\ Speech\ BERT\ App`
  * `python eval.py`
  * http://localhost:3000/
  
* XGBoost App Service
  * `cd Hate\ Speech\ App\ XGBoost`
  * `python app.py`
  * http://localhost:6000/


References - 
1. [Deep Learning for Hate Speech Detection in Tweets](https://arxiv.org/pdf/1706.00188v1.pdf)
2. [Are You a Racist or Am I Seeing Things? Annotator Influence on Hate Speech Detection on Twitter](https://www.aclweb.org/anthology/W16-5618)
3. [Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter](https://www.aclweb.org/anthology/N16-2013)


