import os
from data_model import *

from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEndpoint
#pip install --upgrade langchain-xai


class AIProviderFactory:
    """Factory class to create AI provider instances"""
    
    # Pre-configured model configurations
    MODEL_CONFIGS = {
        # OpenAI Models
        "gpt-4": ModelConfig(
            provider=AIProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.7,
            api_key_env_var="OPENAI_API_KEY"
        ),
        "gpt-3.5-turbo": ModelConfig(
            provider=AIProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            api_key_env_var="OPENAI_API_KEY"
        ),
        
        # Google Gemini Models
        "gemini-pro": ModelConfig(
            provider=AIProvider.GEMINI,
            model_name="gemini-pro",
            temperature=0.7,
            api_key_env_var="GOOGLE_API_KEY"
        ),
        "gemini-1.5-pro": ModelConfig(
            provider=AIProvider.GEMINI,
            model_name="gemini-1.5-pro",
            temperature=0.7,
            api_key_env_var=os.getenv("GEMINI_API_KEY")
        ),
        
        # Claude Models
        "claude-3-opus": ModelConfig(
            provider=AIProvider.CLAUDE,
            model_name="claude-3-opus-20240229",
            temperature=0.7,
            api_key_env_var="ANTHROPIC_API_KEY"
        ),
        "claude-3-sonnet": ModelConfig(
            provider=AIProvider.CLAUDE,
            model_name="claude-3-sonnet-20240229",
            temperature=0.7,
            api_key_env_var="ANTHROPIC_API_KEY"
        ),
        "claude-3-haiku": ModelConfig(
            provider=AIProvider.CLAUDE,
            model_name="claude-3-haiku-20240307",
            temperature=0.7,
            api_key_env_var="ANTHROPIC_API_KEY"
        ),
        
        # HuggingFace Models
        "mistral-7b": ModelConfig(
            provider=AIProvider.HUGGINGFACE,
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            temperature=0.7,
            max_tokens=512,
            api_key_env_var="HUGGINGFACE_API_TOKEN"
        ),
        "llama2-7b": ModelConfig(
            provider=AIProvider.HUGGINGFACE,
            model_name="meta-llama/Llama-2-7b-chat-hf",
            temperature=0.7,
            max_tokens=512,
            api_key_env_var="HUGGINGFACE_API_TOKEN"
        ),
    }
    
    @staticmethod
    def create_llm(model_config: Union[str, ModelConfig]) -> LLM:
        """Create LLM instance based on model configuration"""
        
        if isinstance(model_config, str):
            if model_config not in AIProviderFactory.MODEL_CONFIGS:
                raise ValueError(f"Unknown model configuration: {model_config}")
            config = AIProviderFactory.MODEL_CONFIGS[model_config]
        else:
            config = model_config
        
        # Get API key
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {config.api_key_env_var}")
        
        # Create provider-specific LLM
        if config.provider == AIProvider.OPENAI:
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                openai_api_key=api_key,
                max_tokens=config.max_tokens
            )
        
        elif config.provider == AIProvider.GEMINI:
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                google_api_key=api_key,
                max_output_tokens=config.max_tokens
            )
        
        elif config.provider == AIProvider.CLAUDE:
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                anthropic_api_key=api_key,
                max_tokens=config.max_tokens or 1024
            )
        
        elif config.provider == AIProvider.HUGGINGFACE:
            return HuggingFaceEndpoint(
                repo_id=config.model_name,
                temperature=config.temperature,
                huggingfacehub_api_token=api_key,
                max_length=config.max_tokens or 512,
                **(config.additional_params or {})
            )
        
        else:
            raise ValueError(f"Unsupported AI provider: {config.provider}")
    
    @staticmethod
    def list_available_models() -> Dict[str, ModelConfig]:
        """List all available model configurations"""
        return AIProviderFactory.MODEL_CONFIGS.copy()