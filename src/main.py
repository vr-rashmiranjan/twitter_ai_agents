import asyncio
import sys

from api_provider import AIProviderFactory
from data_model import AIProvider, ModelConfig
from twitter_ai_systems import TwitterAISystem

def list_available_models():
    print("\n Available AI Models:")
    print("=" * 50)
    
    models = AIProviderFactory.list_available_models()
    
    for provider in AIProvider:
        provider_models = {k: v for k, v in models.items() if v.provider == provider}
        if provider_models:
            print(f"\n{provider.value.upper()} Models:")
            for model_name, config in provider_models.items():
                print(f"  - {model_name}")
                print(f"    Model: {config.model_name}")
                print(f"    Temperature: {config.temperature}")
                print(f"    API Key: {config.api_key_env_var}")
                if config.max_tokens:
                    print(f"    Max Tokens: {config.max_tokens}")
    
    print(f"\n Available Image Providers:")
    print("  - openai (DALL-E 3)")
    print("  - stability (Stable Diffusion XL)")
    print("  - huggingface (Stable Diffusion)")
    
    print(f"\nExample Configurations:")
    for config_name, config in EXAMPLE_CONFIGURATIONS.items():
        print(f"  - {config_name}")
        
async def run_with_custom_config():
    """Example of running with completely custom configuration"""
    custom_model = ModelConfig(
        provider=AIProvider.GEMINI,
        model_name="gemini-1.5-pro",
        temperature=0.8,
        api_key_env_var="GOOGLE_API_KEY",
        max_tokens=1024
    )
    twitter_ai = TwitterAISystem(
        trend_analysis_model=custom_model,
        content_research_model="claude-3-sonnet",
        content_creation_model="gpt-4",
        image_provider="stability"
    )
    
    await twitter_ai.run_full_pipeline(posting_interval_hours=6)


if __name__ == "__main__":
    EXAMPLE_CONFIGURATIONS = {
    "openai_only": {
        "trend_analysis_model": "gpt-4.1",
        "content_research_model": "gpt-4.1",
        "content_creation_model": "gpt-4.1",
        "image_provider": "openai"
    },
    
    "open_source": {
        "trend_analysis_model": "mistral-7b",
        "content_research_model": "llama2-7b",
        "content_creation_model": "mistral-7b",
        "image_provider": "huggingface"
    }
}
    if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == "list-models":
                list_available_models()
            elif command == "custom":
                asyncio.run(run_with_custom_config())
            elif command in EXAMPLE_CONFIGURATIONS:
                asyncio.run(main(command))
            else:
                print(f"Unknown command: {command}")
                print("Available commands:")
                print("  list-models    - List all available models")
                print("  custom         - Run with custom configuration")
                print("  Configuration names:")
                for name in EXAMPLE_CONFIGURATIONS:
                    print(f"    {name}")
    else:
        # Default configuration
        pass
        # asyncio.run(main())