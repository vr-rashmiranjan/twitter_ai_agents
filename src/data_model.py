from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

class AIProvider(Enum):
    """Enumeration of supported AI providers"""
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    HUGGINGFACE = "huggingface"


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    provider: AIProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key_env_var: str = ""
    additional_params: Dict[str, Any] = None


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

