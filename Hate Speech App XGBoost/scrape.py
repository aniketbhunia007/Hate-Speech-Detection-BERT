import tweepy
import csv
import pandas as pd
import os
def scrape(hashtag, date, no_of_tweets):
    ####input your credentials here
    
    consumer_key = ## Your Key
    consumer_secret = ## Your Key
    access_token = ## Your Key
    access_token_secret = ## Your Key

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)

    #####

    # Open/Create a file to append data
    # csvFile = open('racist_hashtag.csv', 'a')
    try :
        os.remove('scraped.csv')
        print('file removed')
    except FileNotFoundError:
        print('file does not exists!')
    
    csvFile = open('scraped.csv', 'a')

    #Use csv Writer
    csvWriter = csv.writer(csvFile)

    hashtag = hashtag


    for tweet in tweepy.Cursor(api.search,q=hashtag,
                            lang="en",
                            since=date,tweet_mode='extended').items(no_of_tweets):
        print (tweet.created_at, tweet.full_text)
        
    #     we use tweet_mode='extended' to get the full tweet. And similarly, use the attribute full_text to extract it from the status
        csvWriter.writerow([tweet.created_at, tweet.full_text.encode('utf-8')])


if __name__ == '__main__':
	scrape()
