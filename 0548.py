"""
Project 548: Advanced Response Generation for Chatbots
Description:
Modern chatbot implementation using state-of-the-art transformer models with advanced features
including context awareness, sentiment analysis, response ranking, and conversation memory.
"""

import os
import json
import logging
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    set_seed
)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from streamlit_chat import message

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Data class for chat messages"""
    user_id: str
    message: str
    response: str
    timestamp: datetime
    sentiment: Optional[str] = None
    context_score: Optional[float] = None

class MockDatabase:
    """Mock database for storing conversation history and user data"""
    
    def __init__(self, db_path: str = "chatbot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sentiment TEXT,
                context_score REAL
            )
        ''')
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                preferences TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def save_conversation(self, chat_message: ChatMessage):
        """Save a conversation to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (user_id, message, response, timestamp, sentiment, context_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            chat_message.user_id,
            chat_message.message,
            chat_message.response,
            chat_message.timestamp,
            chat_message.sentiment,
            chat_message.context_score
        ))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve conversation history for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT message, response, timestamp, sentiment, context_score
            FROM conversations
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'message': row[0],
                'response': row[1],
                'timestamp': row[2],
                'sentiment': row[3],
                'context_score': row[4]
            }
            for row in results
        ]

class AdvancedChatbot:
    """Advanced chatbot with modern NLP techniques"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.sentiment_analyzer = None
        self.vectorizer = None
        self.db = MockDatabase()
        
        # Load models
        self._load_models()
        
        # Set random seed for reproducibility
        set_seed(42)
    
    def _load_models(self):
        """Load all required models"""
        try:
            # Load main conversation model
            logger.info(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load sentiment analysis pipeline
            logger.info("Loading sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            
            # Initialize TF-IDF vectorizer for context similarity
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of the input text"""
        try:
            result = self.sentiment_analyzer(text)
            return result[0]['label'].lower()
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return "neutral"
    
    def calculate_context_score(self, current_message: str, history: List[Dict]) -> float:
        """Calculate context relevance score based on conversation history"""
        if not history:
            return 0.0
        
        try:
            # Extract previous messages
            previous_messages = [msg['message'] for msg in history[:5]]  # Last 5 messages
            previous_messages.append(current_message)
            
            # Fit vectorizer and transform
            tfidf_matrix = self.vectorizer.fit_transform(previous_messages)
            
            # Calculate cosine similarity between current message and previous messages
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
            
            return float(np.mean(similarities))
            
        except Exception as e:
            logger.warning(f"Context scoring failed: {e}")
            return 0.0
    
    def generate_response(self, user_input: str, user_id: str = "default", 
                        max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate response using the loaded model with context awareness"""
        try:
            # Get conversation history
            history = self.db.get_conversation_history(user_id, limit=5)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(user_input)
            
            # Calculate context score
            context_score = self.calculate_context_score(user_input, history)
            
            # Prepare input with context
            if history:
                # Use recent conversation as context
                context = " ".join([msg['message'] + " " + msg['response'] 
                                  for msg in reversed(history[-3:])])
                full_input = f"{context} Human: {user_input} Bot:"
            else:
                full_input = f"Human: {user_input} Bot:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(full_input, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            if response.startswith("Human:") or response.startswith("Bot:"):
                response = response.split("Human:")[0].split("Bot:")[0].strip()
            
            # Create chat message object
            chat_message = ChatMessage(
                user_id=user_id,
                message=user_input,
                response=response,
                timestamp=datetime.now(),
                sentiment=sentiment,
                context_score=context_score
            )
            
            # Save to database
            self.db.save_conversation(chat_message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Could you please try again?"
    
    def get_conversation_stats(self, user_id: str) -> Dict:
        """Get conversation statistics for a user"""
        history = self.db.get_conversation_history(user_id, limit=100)
        
        if not history:
            return {"total_messages": 0, "avg_sentiment": "neutral", "avg_context_score": 0.0}
        
        sentiments = [msg['sentiment'] for msg in history if msg['sentiment']]
        context_scores = [msg['context_score'] for msg in history if msg['context_score'] is not None]
        
        return {
            "total_messages": len(history),
            "avg_sentiment": max(set(sentiments), key=sentiments.count) if sentiments else "neutral",
            "avg_context_score": np.mean(context_scores) if context_scores else 0.0
        }

def main():
    """Main function for command-line interface"""
    print("🤖 Advanced Chatbot - Response Generation System")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = AdvancedChatbot()
    
    print("Chatbot initialized! Type 'quit' to exit.")
    print()
    
    user_id = "cli_user"
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue
            
            print("Bot: ", end="", flush=True)
            response = chatbot.generate_response(user_input, user_id)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\nBot: Goodbye! Have a great day!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
