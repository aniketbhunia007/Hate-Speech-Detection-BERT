import tweepy #https://github.com/tweepy/tweepy
import csv
import os

def get_all_tweets(screen_name, no_of_tweets):
    #Twitter API credentials
#     consumer_key = ## Your Key
#     consumer_secret = ## Your Key
#     access_token = ## Your Key
#     access_token_secret = ## Your Key

    consumer_key = 'hQSfuYsO9ybVNVKPCxURndKQ6'
    consumer_secret = 'BUbTUcCSOozzU3PzFSbt44A4nWDjWVccAwuyPTgzQo25ImhcsV'
    access_token = '1030700072755855361-1bl3CyvIAvr8VQ74j368130fq7odUT'
    access_token_secret = '1cw58KNSxH0VCs0bqZCbYdevCIm6sIpzQLkuBvglqirwQ'

    try:
        os.remove('username_tweets.csv')
        print('file removed')
    except FileNotFoundError:
        print('file does not exists!')
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
	
	#initialize a list to hold all the tweepy Tweets
    alltweets = []	
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
  #     we use tweet_mode='extended' to get the full tweet. And similarly, use the attribute full_text to extract it from the status
  
    new_tweets = api.user_timeline(screen_name = screen_name,count=no_of_tweets, tweet_mode='extended')
	
	#save most recent tweets
    alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > no_of_tweets:
        print("getting tweets before %s" % (oldest))
		
		#all subsiquent requests use the max_id param to prevent duplicates
    #     we use tweet_mode='extended' to get the full tweet. And similarly, use the attribute full_text to extract it from the status
    
        new_tweets = api.user_timeline(screen_name = screen_name,max_id=oldest, tweet_mode='extended')
		
		#save most recent tweets
        alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print("...%s tweets downloaded so far" % (len(alltweets)))
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
    outtweets = [[tweet.id_str, tweet.created_at, tweet.full_text.encode("utf-8")] for tweet in alltweets]
       
	#write the csv	
    with open('username_tweets.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(outtweets)
        pass


if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets('realDonaldTrump',30)