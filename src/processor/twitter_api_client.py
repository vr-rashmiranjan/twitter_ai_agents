import logging
import os

import tweepy

from dotenv import load_dotenv

load_dotenv()

class TwitterAPIClient:
    """Twitter API client wrapper"""
    
    def __init__(self):
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        # Initialize Tweepy client
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            wait_on_rate_limit=True
        )
        
        # Initialize API v1.1 for media upload
        auth = tweepy.OAuth1UserHandler(
            self.api_key, self.api_secret,
            self.access_token, self.access_token_secret
        )
        self.api_v1 = tweepy.API(auth)

    def get_trending_keywords(self, woeid: int = 1):
        """
        Get trending keywords/hashtags
        
        Args:
            woeid: Where On Earth ID (1 = Worldwide, 23424977 = United States)
        
        Returns:
            List of TrendingKeyword objects
        """
        try:
            trends = self.api_v1.get_place_trends(woeid)
            return trends
            
        except tweepy.TweepyException as e:
            logging.error(f"Error fetching trending keywords: {e}")
            return []    

if __name__ == "__main__":
    obj = TwitterAPIClient().get_trending_keywords(1)