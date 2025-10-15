#!/bin/bash
# SOTA AP Policy Co-Pilot System Launcher
# Single script to start the complete state-of-the-art system

echo "🚀 AP Policy Co-Pilot - SOTA System"
echo "=================================="

# Check Python virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/update requirements
echo "📦 Installing requirements..."
pip install -q -r requirements.txt

# Check for environment variables
echo "🔍 Checking environment..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not set (Gemini features will be disabled)"
    echo "   Get your key from: https://makersuite.google.com/app/apikey"
    echo "   Set it with: export GEMINI_API_KEY='your_key_here'"
fi

# Check if Weaviate and Neo4j are running
echo "🔧 Checking services..."

# Start Weaviate if not running
if ! curl -s http://localhost:8080/v1/.well-known/ready > /dev/null; then
    echo "🚀 Starting Weaviate..."
    docker-compose up -d weaviate
    echo "⏳ Waiting for Weaviate to start..."
    sleep 15
fi

# Check Neo4j
if ! curl -s http://localhost:7474 > /dev/null; then
    echo "🚀 Starting Neo4j..."
    docker-compose up -d neo4j
    echo "⏳ Waiting for Neo4j to start..."
    sleep 10
fi

echo ""
echo "🎯 Choose how to start the SOTA system:"
echo ""
echo "1. API Server only:"
echo "   python sota_orchestrator.py --action api"
echo ""
echo "2. Dashboard only:"  
echo "   streamlit run sota_dashboard.py"
echo ""
echo "3. System check:"
echo "   python sota_orchestrator.py --action check"
echo ""
echo "4. Search from command line:"
echo "   python sota_orchestrator.py --action search --query 'your query'"
echo ""

read -p "Enter choice (1-4) or press Enter to start API server: " choice

case $choice in
    1|"")
        echo "🌐 Starting API server..."
        python sota_orchestrator.py --action api
        ;;
    2)
        echo "📊 Starting dashboard..."
        streamlit run sota_dashboard.py --server.port 8501
        ;;
    3)
        echo "🔧 Running system check..."
        python sota_orchestrator.py --action check
        ;;
    4)
        read -p "Enter your search query: " query
        echo "🔍 Searching..."
        python sota_orchestrator.py --action search --query "$query"
        ;;
    *)
        echo "❌ Invalid choice. Starting API server by default..."
        python sota_orchestrator.py --action api
        ;;
esac

echo ""
echo "📋 Quick Access URLs:"
echo "   API Server: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   Dashboard: http://localhost:8501"
echo "   Weaviate: http://localhost:8080"
echo "   Neo4j Browser: http://localhost:7474"