# Advanced Chatbot - Quick Start Guide

## 🚀 Quick Start Commands

### 1. Setup (First time only)
```bash
python setup.py
```

### 2. Run Web Interface (Recommended)
```bash
streamlit run app.py
```
Then open: http://localhost:8501

### 3. Run Command Line Interface
```bash
python 0548.py
```

### 4. Run Tests
```bash
pytest test_chatbot.py -v
```

## 🔧 Configuration

Edit `.env` file to customize:
- Model selection
- Response length
- Temperature (creativity)
- Database settings

## 📊 Features

✅ **Context-Aware Responses** - Maintains conversation history  
✅ **Sentiment Analysis** - Analyzes user emotions  
✅ **Web Interface** - Beautiful Streamlit UI  
✅ **Database Storage** - Persistent conversation history  
✅ **GPU Support** - Automatic CUDA acceleration  
✅ **Comprehensive Testing** - Full test suite included  

## 🆘 Troubleshooting

**Model loading issues?**
```bash
rm -rf ~/.cache/huggingface/
```

**CUDA out of memory?**
Set `DEVICE=cpu` in `.env` file

**Database errors?**
```bash
rm chatbot.db
```

## 📚 Documentation

See `README.md` for complete documentation.

---
**Happy Chatting! 🤖💬**
