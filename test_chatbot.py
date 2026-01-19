"""
Test suite for the Advanced Chatbot system
"""

import pytest
import tempfile
import os
import sqlite3
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from 0548 import AdvancedChatbot, MockDatabase, ChatMessage
from config import ChatbotConfig

class TestMockDatabase:
    """Test cases for MockDatabase class"""
    
    def setup_method(self):
        """Setup test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = MockDatabase(self.temp_db.name)
    
    def teardown_method(self):
        """Cleanup test database"""
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization"""
        assert os.path.exists(self.temp_db.name)
        
        # Check if tables exist
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'conversations' in tables
        assert 'users' in tables
        
        conn.close()
    
    def test_save_conversation(self):
        """Test saving conversation to database"""
        chat_message = ChatMessage(
            user_id="test_user",
            message="Hello",
            response="Hi there!",
            timestamp=datetime.now(),
            sentiment="positive",
            context_score=0.8
        )
        
        self.db.save_conversation(chat_message)
        
        # Verify data was saved
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM conversations WHERE user_id = ?", ("test_user",))
        result = cursor.fetchone()
        
        assert result is not None
        assert result[1] == "test_user"  # user_id
        assert result[2] == "Hello"      # message
        assert result[3] == "Hi there!" # response
        assert result[5] == "positive"  # sentiment
        assert result[6] == 0.8          # context_score
        
        conn.close()
    
    def test_get_conversation_history(self):
        """Test retrieving conversation history"""
        # Save multiple conversations
        for i in range(5):
            chat_message = ChatMessage(
                user_id="test_user",
                message=f"Message {i}",
                response=f"Response {i}",
                timestamp=datetime.now(),
                sentiment="neutral",
                context_score=0.5
            )
            self.db.save_conversation(chat_message)
        
        history = self.db.get_conversation_history("test_user", limit=3)
        
        assert len(history) == 3
        assert history[0]['message'] == "Message 4"  # Most recent first
        assert history[2]['message'] == "Message 2"

class TestAdvancedChatbot:
    """Test cases for AdvancedChatbot class"""
    
    @patch('0548.AutoTokenizer')
    @patch('0548.AutoModelForCausalLM')
    @patch('0548.pipeline')
    def setup_method(self, mock_pipeline, mock_model, mock_tokenizer):
        """Setup test chatbot with mocked models"""
        # Mock the model loading
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Mock the database
        with patch('0548.MockDatabase') as mock_db_class:
            mock_db_instance = Mock()
            mock_db_class.return_value = mock_db_instance
            mock_db_instance.get_conversation_history.return_value = []
            
            self.chatbot = AdvancedChatbot()
            self.chatbot.db = mock_db_instance
    
    def teardown_method(self):
        """Cleanup test files"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        # Mock sentiment analyzer
        self.chatbot.sentiment_analyzer = Mock()
        self.chatbot.sentiment_analyzer.return_value = [{'label': 'POSITIVE'}]
        
        sentiment = self.chatbot.analyze_sentiment("I love this!")
        assert sentiment == "positive"
    
    def test_context_score_calculation(self):
        """Test context score calculation"""
        # Mock vectorizer
        self.chatbot.vectorizer = Mock()
        mock_matrix = Mock()
        self.chatbot.vectorizer.fit_transform.return_value = mock_matrix
        
        # Mock cosine similarity
        with patch('0548.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.8, 0.6, 0.7]]
            
            history = [
                {'message': 'Hello'},
                {'message': 'How are you?'},
                {'message': 'Good morning'}
            ]
            
            score = self.chatbot.calculate_context_score("Hello there", history)
            assert score == 0.7  # Average of similarities
    
    @patch('0548.torch')
    def test_generate_response(self, mock_torch):
        """Test response generation"""
        # Mock tokenizer and model
        self.chatbot.tokenizer = Mock()
        self.chatbot.tokenizer.encode.return_value = Mock()
        self.chatbot.tokenizer.decode.return_value = "Hello! How can I help you?"
        self.chatbot.tokenizer.eos_token_id = 50256
        
        self.chatbot.model = Mock()
        mock_output = Mock()
        mock_output.shape = [1, 10]
        self.chatbot.model.generate.return_value = mock_output
        
        # Mock sentiment analysis
        self.chatbot.analyze_sentiment = Mock(return_value="positive")
        
        # Mock context scoring
        self.chatbot.calculate_context_score = Mock(return_value=0.8)
        
        response = self.chatbot.generate_response("Hello", "test_user")
        
        assert response == "Hello! How can I help you?"
        assert self.chatbot.db.save_conversation.called
    
    def test_conversation_stats(self):
        """Test conversation statistics"""
        # Mock database history
        mock_history = [
            {'sentiment': 'positive', 'context_score': 0.8},
            {'sentiment': 'negative', 'context_score': 0.6},
            {'sentiment': 'positive', 'context_score': 0.9}
        ]
        
        self.chatbot.db.get_conversation_history.return_value = mock_history
        
        stats = self.chatbot.get_conversation_stats("test_user")
        
        assert stats['total_messages'] == 3
        assert stats['avg_sentiment'] == 'positive'
        assert stats['avg_context_score'] == 0.7666666666666666

class TestChatbotConfig:
    """Test cases for ChatbotConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ChatbotConfig()
        
        assert config.model_name == "microsoft/DialoGPT-medium"
        assert config.max_length == 100
        assert config.temperature == 0.7
        assert config.device == "auto"
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = ChatbotConfig()
        
        # Valid configuration should pass
        assert config.validate() == True
        
        # Test invalid temperature
        config.temperature = 3.0
        with pytest.raises(ValueError):
            config.validate()
        
        # Reset and test invalid max_length
        config.temperature = 0.7
        config.max_length = 5
        with pytest.raises(ValueError):
            config.validate()
        
        # Reset and test invalid device
        config.max_length = 100
        config.device = "invalid"
        with pytest.raises(ValueError):
            config.validate()

class TestIntegration:
    """Integration tests"""
    
    @patch('0548.AutoTokenizer')
    @patch('0548.AutoModelForCausalLM')
    @patch('0548.pipeline')
    def test_full_conversation_flow(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test complete conversation flow"""
        # Setup mocks
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        try:
            with patch('0548.MockDatabase') as mock_db_class:
                mock_db_instance = Mock()
                mock_db_class.return_value = mock_db_instance
                mock_db_instance.get_conversation_history.return_value = []
                
                chatbot = AdvancedChatbot()
                chatbot.db = mock_db_instance
                
                # Mock response generation
                chatbot.tokenizer = Mock()
                chatbot.tokenizer.encode.return_value = Mock()
                chatbot.tokenizer.decode.return_value = "Hello! How can I help you today?"
                chatbot.tokenizer.eos_token_id = 50256
                
                chatbot.model = Mock()
                mock_output = Mock()
                mock_output.shape = [1, 10]
                chatbot.model.generate.return_value = mock_output
                
                chatbot.analyze_sentiment = Mock(return_value="positive")
                chatbot.calculate_context_score = Mock(return_value=0.8)
                
                # Test conversation
                response = chatbot.generate_response("Hello", "test_user")
                
                assert response == "Hello! How can I help you today?"
                assert chatbot.db.save_conversation.called
                
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)

# Pytest fixtures
@pytest.fixture
def temp_database():
    """Fixture for temporary database"""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    db = MockDatabase(temp_db.name)
    yield db
    os.unlink(temp_db.name)

@pytest.fixture
def sample_chat_message():
    """Fixture for sample chat message"""
    return ChatMessage(
        user_id="test_user",
        message="Hello, how are you?",
        response="I'm doing well, thank you!",
        timestamp=datetime.now(),
        sentiment="positive",
        context_score=0.8
    )

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
