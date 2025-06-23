import sys
from pathlib import Path

current_file = Path(__file__).resolve()
testfiles_path = current_file.parent.parent / 'src'
sys.path.insert(0, str(testfiles_path))
#-------------------------------------------------------------#
from processor.twitter_api_client import TwitterAPIClient
from tools.trending_keyword_tool import TrendingKeywordsTool

twitter_client = TwitterAPIClient
trending_tool = TrendingKeywordsTool(twitter_client)

trending_tool._run()

# import os
# import sys

# print("Current file:", __file__)
# print("Current directory:", os.path.dirname(os.path.abspath(__file__)))
# print("Python path:", sys.path)

# # Check if your target directory exists
# testfiles_path = "/Users/arundhatibhowmick/Desktop/python_projects/twitter_ai_agents/src/processor/twitter_api_client.py"  # adjust this
# print("Testfiles exists:", os.path.exists(testfiles_path))
# # print("Testfiles contents:", os.listdir(testfiles_path) if os.path.exists(testfiles_path) else "Not found")

