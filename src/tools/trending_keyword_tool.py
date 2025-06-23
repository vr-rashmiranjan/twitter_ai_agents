from pydantic import Field
from typing import List, Optional, Any
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from data_model import TrendingKeyword
from processor.twitter_api_client import TwitterAPIClient

class TrendingKeywordsTool(BaseTool):
    """Tool to get trending keywords from Twitter"""
    
    name :str = "trending_keywords"
    description : str = "Get top trending keywords/hashtags from Twitter for the last 7 days"
    twitter_client: str = Field(default=None,exclude=True) 
    
    def __init__(self, twitter_client: TwitterAPIClient()):
        super().__init__()
        self.__twitter_client = twitter_client
    
    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[TrendingKeyword]:
        """Get trending keywords"""
        try:
            # Get trending topics (worldwide)
            twitter_client = self.__twitter_client
            trends = twitter_client.get_trending_keywords(1)
            
            trending_keywords = []
            for i, trend in enumerate(trends[0].trends[:10]):  # Top 10 trends
                keyword = TrendingKeyword(
                    keyword=trend.name,
                    volume=trend.tweet_volume or 0,
                    rank=i + 1
                )
                trending_keywords.append(keyword)
            
            return trending_keywords
        
        except Exception as e:
            print(f"Error getting trending keywords: {e}")
            return []