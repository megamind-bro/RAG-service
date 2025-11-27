# 🚀 Quick Start Guide

Get your RAG service up and running in 5 minutes!

## Prerequisites

- ✅ Python 3.8+
- ✅ Ollama installed
- ✅ Google API Key

## Step 1: Install Ollama

### macOS/Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download from: https://ollama.com/download

## Step 2: Setup Project

```bash
# Clone and enter directory
cd rag-service

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
```

## Step 3: Configure Environment

Edit `.env` file:
```env
GOOGLE_API_KEY=your_actual_google_api_key_here
```

Get your Google API key from: https://makersuite.google.com/app/apikey

## Step 4: Start Services

### Terminal 1: Start Ollama
```bash
ollama serve
ollama pull nomic-embed-text
```

### Terminal 2: Start RAG Service
```bash
uvicorn app.main:app --reload --port 8088
```

## Step 5: Test It!

Open your browser: http://localhost:8088/docs

### Try the API

```bash
# Health check
curl http://localhost:8088/health

# Ask a question
curl -X POST "http://localhost:8088/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the best soil pH for maize?",
    "user_id": "test_user"
  }'
```

## Step 6: Ingest Documents (Optional)

```bash
# Upload a PDF or text file
curl -X POST "http://localhost:8088/ingest/files" \
  -F "files=@your_document.pdf"
```

## 🎉 You're Ready!

Your RAG service is now running at:
- **API**: http://localhost:8088
- **Docs**: http://localhost:8088/docs
- **ReDoc**: http://localhost:8088/redoc

## Next Steps

1. **Read Full Documentation**: [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
2. **Integration Examples**: [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)
3. **Deploy to Production**: See [README_NEW.md](./README_NEW.md)

## Common Issues

### "Ollama connection refused"
```bash
# Make sure Ollama is running
ollama serve
```

### "Google API error"
```bash
# Check your API key in .env
cat .env | grep GOOGLE_API_KEY
```

### "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

## Test Endpoints

### Create a conversation
```bash
curl -X POST "http://localhost:8088/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "title": "My First Conversation"
  }'
```

### Get user conversations
```bash
curl "http://localhost:8088/users/test_user/conversations"
```

### Query with conversation history
```bash
curl -X POST "http://localhost:8088/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I prepare soil for maize?",
    "user_id": "test_user",
    "conversation_id": "your-conversation-id-here"
  }'
```

## Docker Quick Start

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## Need Help?

- 📖 Full API Docs: [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
- 💻 Integration Guide: [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)
- 🐛 Issues: Create a GitHub issue

---

**Happy Coding! 🌽**
