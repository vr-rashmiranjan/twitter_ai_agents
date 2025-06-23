from typing import Union
from data_model import ModelConfig
from processor.twitter_api_client import TwitterAPIClient


class TwitterAISystem:
    """Main orchestrator for the Twitter AI system"""
    
    def __init__(self, 
            trend_analysis_model: Union[str, ModelConfig] = "gpt-4",
            content_research_model: Union[str, ModelConfig] = "gpt-4", 
            content_creation_model: Union[str, ModelConfig] = "gpt-4",
            image_provider: str = "openai"):
        
        self.twitter_client = TwitterAPIClient()
        self.trend_agent = TrendAnalysisAgent(self.twitter_client, trend_analysis_model)
        self.research_agent = ContentResearchAgent(self.twitter_client, content_research_model)
        self.content_agent = ContentCreationAgent(content_creation_model)
        self.posting_agent = PostingAgent(self.twitter_client, image_provider)
    
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