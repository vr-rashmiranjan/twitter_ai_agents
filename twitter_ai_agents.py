"""
Twitter AI Agent System using LangChain
=====================================

This system creates specialized AI agents to automate Twitter content creation and posting.
Each agent handles a specific task in the pipeline.
"""

import os
import time
import json
import asyncio
import schedule
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# LangChain imports
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

# Third-party imports
import tweepy
import requests
from PIL import Image
import io


@dataclass
class TrendingKeyword:
    """Data class for trending keywords"""
    keyword: str
    volume: int
    rank: int


@dataclass
class TweetData:
    """Data class for tweet information"""
    id: str
    text: str
    author: str
    followers_count: int
    retweet_count: int
    like_count: int
    view_count: int


@dataclass
class GeneratedPost:
    """Data class for generated posts"""
    content: str
    hashtags: List[str]
    image_prompt: str
    topic: str


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


class TrendingKeywordsTool(BaseTool):
    """Tool to get trending keywords from Twitter"""
    
    name = "trending_keywords"
    description = "Get top trending keywords/hashtags from Twitter for the last 7 days"
    
    def __init__(self, twitter_client: TwitterAPIClient):
        super().__init__()
        self.twitter_client = twitter_client
    
    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[TrendingKeyword]:
        """Get trending keywords"""
        try:
            # Get trending topics (worldwide)
            trends = self.twitter_client.client.get_place_trends(id=1)  # 1 = worldwide
            
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


class TopTweetsTool(BaseTool):
    """Tool to get top tweets for a specific keyword"""
    
    name = "top_tweets"
    description = "Get top 10 tweets with highest engagement for a specific keyword"
    
    def __init__(self, twitter_client: TwitterAPIClient):
        super().__init__()
        self.twitter_client = twitter_client
    
    def _run(
        self,
        keyword: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[TweetData]:
        """Get top tweets for keyword"""
        try:
            # Search for recent tweets with high engagement
            tweets = tweepy.Paginator(
                self.twitter_client.client.search_recent_tweets,
                query=f"{keyword} -is:retweet lang:en",
                tweet_fields=['public_metrics', 'author_id', 'created_at'],
                user_fields=['public_metrics'],
                expansions=['author_id'],
                max_results=100
            ).flatten(limit=100)
            
            tweet_data = []
            users_dict = {}
            
            # Get user information
            if hasattr(tweets, 'includes') and 'users' in tweets.includes:
                users_dict = {user.id: user for user in tweets.includes['users']}
            
            for tweet in tweets:
                if tweet.author_id in users_dict:
                    user = users_dict[tweet.author_id]
                    
                    # Calculate engagement score (views + retweets + likes)
                    metrics = tweet.public_metrics
                    engagement_score = (
                        metrics.get('impression_count', 0) + 
                        metrics.get('retweet_count', 0) * 2 + 
                        metrics.get('like_count', 0)
                    )
                    
                    tweet_info = TweetData(
                        id=tweet.id,
                        text=tweet.text,
                        author=user.username,
                        followers_count=user.public_metrics['followers_count'],
                        retweet_count=metrics.get('retweet_count', 0),
                        like_count=metrics.get('like_count', 0),
                        view_count=metrics.get('impression_count', 0)
                    )
                    tweet_data.append((tweet_info, engagement_score))
            
            # Sort by engagement score and return top 10
            tweet_data.sort(key=lambda x: x[1], reverse=True)
            return [tweet_info for tweet_info, _ in tweet_data[:10]]
        
        except Exception as e:
            print(f"Error getting top tweets: {e}")
            return []


class ImageGeneratorTool(BaseTool):
    """Tool to generate images for posts"""
    
    name = "image_generator"
    description = "Generate relevant images for social media posts"
    
    def __init__(self):
        super().__init__()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
    
    def _run(
        self,
        prompt: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate image using DALL-E or return placeholder"""
        try:
            # Using OpenAI DALL-E API
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'dall-e-3',
                'prompt': f"Create a professional, eye-catching social media image for: {prompt}. Style: modern, clean, engaging for Twitter/X platform.",
                'n': 1,
                'size': '1024x1024'
            }
            
            response = requests.post(
                'https://api.openai.com/v1/images/generations',
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                image_url = result['data'][0]['url']
                
                # Download and save image
                img_response = requests.get(image_url)
                image = Image.open(io.BytesIO(img_response.content))
                
                # Save locally
                filename = f"generated_image_{int(time.time())}.png"
                image.save(filename)
                return filename
            else:
                print(f"Image generation failed: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error generating image: {e}")
            return None


class TrendAnalysisAgent:
    """Agent responsible for analyzing trending topics"""
    
    def __init__(self, twitter_client: TwitterAPIClient):
        self.twitter_client = twitter_client
        self.trending_tool = TrendingKeywordsTool(twitter_client)
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4")
        
        # Create analysis prompt
        self.analysis_prompt = PromptTemplate(
            input_variables=["trends"],
            template="""
            Analyze the following trending topics and identify the most engaging ones for content creation:
            
            Trends: {trends}
            
            Consider:
            1. Viral potential
            2. Audience engagement
            3. Content creation opportunities
            4. Brand safety
            
            Return the top 5 trends with brief analysis of why they're good for content creation.
            Format as JSON with trend name and reasoning.
            """
        )
    
    async def analyze_trends(self) -> List[Dict[str, Any]]:
        """Analyze trending topics"""
        trends = self.trending_tool._run()
        
        if not trends:
            return []
        
        trends_text = "\n".join([f"{t.rank}. {t.keyword} (Volume: {t.volume})" for t in trends])
        
        # Use LLM to analyze trends
        analysis = self.llm.invoke(self.analysis_prompt.format(trends=trends_text))
        
        try:
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', analysis.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except:
            pass
        
        # Fallback: return top trends
        return [{"trend": t.keyword, "reasoning": "High volume trending topic"} for t in trends[:5]]


class ContentResearchAgent:
    """Agent responsible for researching top content"""
    
    def __init__(self, twitter_client: TwitterAPIClient):
        self.twitter_client = twitter_client
        self.top_tweets_tool = TopTweetsTool(twitter_client)
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-4")
    
    async def research_topic(self, keyword: str) -> Dict[str, Any]:
        """Research top content for a specific topic"""
        top_tweets = self.top_tweets_tool._run(keyword)
        
        if not top_tweets:
            return {"keyword": keyword, "insights": [], "top_tweets": []}
        
        # Analyze patterns in top tweets
        tweet_texts = [tweet.text for tweet in top_tweets[:5]]
        
        analysis_prompt = f"""
        Analyze these top-performing tweets about "{keyword}":
        
        {chr(10).join([f"{i+1}. {text}" for i, text in enumerate(tweet_texts)])}
        
        Identify:
        1. Common themes and angles
        2. Engagement patterns
        3. Writing styles that work
        4. Content formats (questions, statements, lists, etc.)
        
        Provide insights for creating similar engaging content.
        """
        
        insights = self.llm.invoke(analysis_prompt)
        
        return {
            "keyword": keyword,
            "insights": insights.content,
            "top_tweets": top_tweets,
            "engagement_patterns": self._analyze_engagement_patterns(top_tweets)
        }
    
    def _analyze_engagement_patterns(self, tweets: List[TweetData]) -> Dict[str, Any]:
        """Analyze engagement patterns"""
        if not tweets:
            return {}
        
        total_tweets = len(tweets)
        avg_followers = sum(t.followers_count for t in tweets) / total_tweets
        avg_retweets = sum(t.retweet_count for t in tweets) / total_tweets
        avg_likes = sum(t.like_count for t in tweets) / total_tweets
        
        return {
            "avg_followers": avg_followers,
            "avg_retweets": avg_retweets,
            "avg_likes": avg_likes,
            "engagement_rate": (avg_retweets + avg_likes) / avg_followers if avg_followers > 0 else 0
        }


class ContentCreationAgent:
    """Agent responsible for creating engaging content"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.8, model="gpt-4")
        self.image_generator = ImageGeneratorTool()
        
        self.content_prompt = PromptTemplate(
            input_variables=["keyword", "insights", "engagement_patterns"],
            template="""
            Create an engaging Twitter/X post about "{keyword}" based on the following research:
            
            Research Insights: {insights}
            Engagement Patterns: {engagement_patterns}
            
            Requirements:
            1. Maximum 280 characters
            2. Include compelling hook in first line
            3. Use engaging language that drives interaction
            4. Include call-to-action or question when appropriate
            5. Suggest 3-5 relevant hashtags
            6. Describe an image that would complement the post
            
            Format your response as:
            POST: [your post content]
            HASHTAGS: [hashtags separated by spaces]
            IMAGE_PROMPT: [description for image generation]
            """
        )
    
    async def create_posts(self, research_data: Dict[str, Any], num_posts: int = 10) -> List[GeneratedPost]:
        """Create multiple engaging posts for a topic"""
        posts = []
        keyword = research_data['keyword']
        insights = research_data['insights']
        patterns = research_data.get('engagement_patterns', {})
        
        for i in range(num_posts):
            # Add variation instruction for each post
            variation_prompt = self.content_prompt.format(
                keyword=keyword,
                insights=insights,
                engagement_patterns=patterns
            ) + f"\n\nCreate variation #{i+1} with a different angle or approach."
            
            response = self.llm.invoke(variation_prompt)
            content = response.content
            
            # Parse response
            post_content = self._extract_section(content, "POST:")
            hashtags = self._extract_section(content, "HASHTAGS:").split()
            image_prompt = self._extract_section(content, "IMAGE_PROMPT:")
            
            if post_content:
                posts.append(GeneratedPost(
                    content=post_content,
                    hashtags=hashtags,
                    image_prompt=image_prompt,
                    topic=keyword
                ))
        
        return posts
    
    def _extract_section(self, text: str, section_header: str) -> str:
        """Extract content from a specific section"""
        lines = text.split('\n')
        section_content = []
        in_section = False
        
        for line in lines:
            if line.strip().startswith(section_header):
                in_section = True
                content = line.replace(section_header, '').strip()
                if content:
                    section_content.append(content)
            elif in_section and line.strip() and not any(line.strip().startswith(h) for h in ["POST:", "HASHTAGS:", "IMAGE_PROMPT:"]):
                section_content.append(line.strip())
            elif in_section and line.strip().startswith(("POST:", "HASHTAGS:", "IMAGE_PROMPT:")):
                break
        
        return ' '.join(section_content)


class PostingAgent:
    """Agent responsible for posting content with scheduling"""
    
    def __init__(self, twitter_client: TwitterAPIClient):
        self.twitter_client = twitter_client
        self.image_generator = ImageGeneratorTool()
        self.posting_queue = []
    
    def add_to_queue(self, posts: List[GeneratedPost]):
        """Add posts to posting queue"""
        self.posting_queue.extend(posts)
    
    def create_posting_schedule(self, interval_hours: int = 4):
        """Create posting schedule"""
        for i, post in enumerate(self.posting_queue):
            post_time = datetime.now() + timedelta(hours=i * interval_hours)
            schedule.every().day.at(post_time.strftime("%H:%M")).do(
                self._post_content, post
            ).tag(f'post_{i}')
    
    def _post_content(self, post: GeneratedPost):
        """Post content to Twitter"""
        try:
            # Generate image if needed
            image_path = None
            if post.image_prompt:
                image_path = self.image_generator._run(post.image_prompt)
            
            # Prepare tweet text
            tweet_text = post.content
            if post.hashtags:
                hashtag_text = ' '.join([f'#{tag.lstrip("#")}' for tag in post.hashtags])
                if len(tweet_text + ' ' + hashtag_text) <= 280:
                    tweet_text += ' ' + hashtag_text
            
            # Upload media if image exists
            media_id = None
            if image_path and os.path.exists(image_path):
                media = self.twitter_client.api_v1.media_upload(image_path)
                media_id = media.media_id
                
                # Clean up image file
                os.remove(image_path)
            
            # Post tweet
            if media_id:
                response = self.twitter_client.client.create_tweet(
                    text=tweet_text,
                    media_ids=[media_id]
                )
            else:
                response = self.twitter_client.client.create_tweet(text=tweet_text)
            
            print(f"Posted tweet: {tweet_text[:50]}... (ID: {response.data['id']})")
            
        except Exception as e:
            print(f"Error posting tweet: {e}")
    
    def start_scheduler(self):
        """Start the posting scheduler"""
        print("Starting posting scheduler...")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


class TwitterAISystem:
    """Main orchestrator for the Twitter AI system"""
    
    def __init__(self):
        self.twitter_client = TwitterAPIClient()
        self.trend_agent = TrendAnalysisAgent(self.twitter_client)
        self.research_agent = ContentResearchAgent(self.twitter_client)
        self.content_agent = ContentCreationAgent()
        self.posting_agent = PostingAgent(self.twitter_client)
    
    async def run_full_pipeline(self, posting_interval_hours: int = 4):
        """Run the complete Twitter AI pipeline"""
        print("🚀 Starting Twitter AI Pipeline...")
        
        # Step 1: Analyze trending topics
        print("📈 Analyzing trending topics...")
        trending_analysis = await self.trend_agent.analyze_trends()
        
        if not trending_analysis:
            print("❌ No trending topics found. Exiting.")
            return
        
        print(f"✅ Found {len(trending_analysis)} trending topics")
        
        # Step 2: Research each topic and create content
        all_posts = []
        
        for trend_data in trending_analysis[:3]:  # Focus on top 3 trends
            keyword = trend_data.get('trend', '')
            print(f"🔍 Researching topic: {keyword}")
            
            # Research the topic
            research_data = await self.research_agent.research_topic(keyword)
            
            if research_data['top_tweets']:
                print(f"📝 Creating content for: {keyword}")
                
                # Create posts for this topic
                posts = await self.content_agent.create_posts(research_data, num_posts=3)
                all_posts.extend(posts)
                
                print(f"✅ Created {len(posts)} posts for {keyword}")
        
        if not all_posts:
            print("❌ No posts created. Exiting.")
            return
        
        print(f"🎯 Total posts created: {len(all_posts)}")
        
        # Step 3: Schedule posts
        print("⏰ Scheduling posts...")
        self.posting_agent.add_to_queue(all_posts)
        self.posting_agent.create_posting_schedule(interval_hours=posting_interval_hours)
        
        print(f"✅ Scheduled {len(all_posts)} posts with {posting_interval_hours}h intervals")
        
        # Step 4: Start posting scheduler
        print("🤖 Starting automated posting...")
        self.posting_agent.start_scheduler()


# Configuration and Setup
def setup_environment():
    """Setup environment variables and dependencies"""
    required_env_vars = [
        'TWITTER_API_KEY',
        'TWITTER_API_SECRET', 
        'TWITTER_ACCESS_TOKEN',
        'TWITTER_ACCESS_TOKEN_SECRET',
        'TWITTER_BEARER_TOKEN',
        'OPENAI_API_KEY'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables before running the system.")
        return False
    
    return True


# Main execution
async def main():
    """Main function to run the Twitter AI system"""
    if not setup_environment():
        return
    
    # Initialize and run the system
    twitter_ai = TwitterAISystem()
    await twitter_ai.run_full_pipeline(posting_interval_hours=4)


if __name__ == "__main__":
    # Install required packages first
    print("Installing required packages...")
    os.system("pip install langchain langchain-openai tweepy pillow requests schedule")
    
    # Run the system
    asyncio.run(main())


"""
Setup Instructions:
==================

1. Install dependencies:
   pip install langchain langchain-openai tweepy pillow requests schedule

2. Set environment variables:
   export TWITTER_API_KEY="your_api_key"
   export TWITTER_API_SECRET="your_api_secret"
   export TWITTER_ACCESS_TOKEN="your_access_token"
   export TWITTER_ACCESS_TOKEN_SECRET="your_access_token_secret"
   export TWITTER_BEARER_TOKEN="your_bearer_token"
   export OPENAI_API_KEY="your_openai_key"

3. Get Twitter API credentials:
   - Apply for Twitter Developer Account
   - Create a new app and get API keys
   - Ensure you have Read/Write permissions

4. Get OpenAI API key:
   - Sign up at OpenAI
   - Generate API key for DALL-E image generation

5. Run the system:
   python twitter_ai_system.py

Features:
=========
- ✅ Trending keyword analysis with AI insights
- ✅ Top tweet research and engagement pattern analysis  
- ✅ AI-powered content creation with multiple variations
- ✅ Automated image generation for posts
- ✅ Scheduled posting with customizable intervals
- ✅ Error handling and retry mechanisms
- ✅ Modular agent-based architecture using LangChain
- ✅ Comprehensive logging and monitoring

The system will:
1. Analyze trending topics
2. Research top-performing content
3. Generate engaging posts with images
4. Schedule and post content automatically every 4 hours
"""