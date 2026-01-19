"""
Streamlit Web UI for Advanced Chatbot
Modern, interactive web interface for the chatbot system
"""

import streamlit as st
import time
import random
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import our chatbot classes
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main chatbot module
import importlib.util
spec = importlib.util.spec_from_file_location("chatbot_module", "0548.py")
chatbot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chatbot_module)

AdvancedChatbot = chatbot_module.AdvancedChatbot
MockDatabase = chatbot_module.MockDatabase

# Page configuration
st.set_page_config(
    page_title="🤖 Advanced Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        margin-right: 20%;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .stButton > button {
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #5a6fd8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot():
    """Load the chatbot model with caching"""
    with st.spinner("🤖 Loading AI models... This may take a moment."):
        try:
            chatbot = AdvancedChatbot()
            return chatbot
        except Exception as e:
            st.error(f"Failed to load chatbot: {e}")
            return None

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{random.randint(1000, 9999)}"
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = load_chatbot()
    if "db" not in st.session_state:
        st.session_state.db = MockDatabase()

def display_chat_message(message, is_user=True):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Bot:</strong> {message}
        </div>
        """, unsafe_allow_html=True)

def display_conversation_stats():
    """Display conversation statistics"""
    if st.session_state.chatbot:
        stats = st.session_state.chatbot.get_conversation_stats(st.session_state.user_id)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Messages", stats["total_messages"])
        
        with col2:
            sentiment_emoji = {
                "positive": "😊",
                "negative": "😔", 
                "neutral": "😐"
            }.get(stats["avg_sentiment"], "😐")
            st.metric("Avg Sentiment", f"{sentiment_emoji} {stats['avg_sentiment']}")
        
        with col3:
            st.metric("Context Score", f"{stats['avg_context_score']:.2f}")

def display_conversation_history():
    """Display conversation history"""
    history = st.session_state.db.get_conversation_history(st.session_state.user_id, limit=20)
    
    if history:
        st.subheader("📜 Recent Conversation History")
        
        # Create a DataFrame for better display
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Display in reverse chronological order
        for _, row in df.iterrows():
            with st.expander(f"💬 {row['timestamp'].strftime('%H:%M:%S')} - {row['message'][:50]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**You:** {row['message']}")
                with col2:
                    st.write(f"**Bot:** {row['response']}")
                
                if row['sentiment']:
                    sentiment_emoji = {"positive": "😊", "negative": "😔", "neutral": "😐"}.get(row['sentiment'], "😐")
                    st.write(f"**Sentiment:** {sentiment_emoji} {row['sentiment']}")
                
                if row['context_score'] is not None:
                    st.write(f"**Context Score:** {row['context_score']:.3f}")

def create_sentiment_chart():
    """Create a sentiment analysis chart"""
    history = st.session_state.db.get_conversation_history(st.session_state.user_id, limit=50)
    
    if len(history) > 1:
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Count sentiments
        sentiment_counts = df['sentiment'].value_counts()
        
        # Create pie chart
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': '#4CAF50',
                'negative': '#F44336',
                'neutral': '#FFC107'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Advanced AI Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1, 
                              help="Controls randomness in responses. Lower = more focused, Higher = more creative")
        max_length = st.slider("Max Response Length", 50, 200, 100, 10,
                              help="Maximum length of generated responses")
        
        # User info
        st.subheader("User Information")
        st.write(f"**User ID:** {st.session_state.user_id}")
        
        # Statistics
        st.subheader("📊 Statistics")
        display_conversation_stats()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", help="Clear all conversation history"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(message["content"], message["role"] == "user")
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message(prompt, True)
            
            # Generate bot response
            if st.session_state.chatbot:
                with st.spinner("🤖 Thinking..."):
                    response = st.session_state.chatbot.generate_response(
                        prompt, 
                        st.session_state.user_id,
                        max_length=max_length,
                        temperature=temperature
                    )
                
                # Add bot response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message(response, False)
            else:
                st.error("Chatbot not loaded. Please refresh the page.")
    
    with col2:
        st.subheader("📈 Analytics")
        
        # Sentiment chart
        create_sentiment_chart()
        
        # Quick stats
        if st.session_state.messages:
            st.metric("Messages in Session", len(st.session_state.messages))
            
            # Response time simulation
            avg_response_time = random.uniform(0.5, 2.0)
            st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Advanced AI • Built with Streamlit • Enhanced with Modern NLP</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
