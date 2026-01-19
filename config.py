"""
Configuration management for the Advanced Chatbot
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ChatbotConfig:
    """Configuration class for the chatbot system"""
    
    # Model settings
    model_name: str = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
    max_length: int = int(os.getenv("MAX_LENGTH", "100"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    device: str = os.getenv("DEVICE", "auto")  # auto, cpu, cuda
    
    # Database settings
    db_path: str = os.getenv("DB_PATH", "chatbot.db")
    
    # Sentiment analysis
    sentiment_model: str = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    # UI settings
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    streamlit_host: str = os.getenv("STREAMLIT_HOST", "localhost")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Performance settings
    use_cache: bool = os.getenv("USE_CACHE", "true").lower() == "true"
    cache_size: int = int(os.getenv("CACHE_SIZE", "100"))
    
    # Security settings
    enable_user_tracking: bool = os.getenv("ENABLE_USER_TRACKING", "true").lower() == "true"
    max_messages_per_session: int = int(os.getenv("MAX_MESSAGES_PER_SESSION", "1000"))
    
    @classmethod
    def from_env(cls) -> "ChatbotConfig":
        """Create configuration from environment variables"""
        return cls()
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        if self.temperature < 0.1 or self.temperature > 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")
        
        if self.max_length < 10 or self.max_length > 500:
            raise ValueError("Max length must be between 10 and 500")
        
        if self.device not in ["auto", "cpu", "cuda"]:
            raise ValueError("Device must be 'auto', 'cpu', or 'cuda'")
        
        return True

# Global configuration instance
config = ChatbotConfig.from_env()
