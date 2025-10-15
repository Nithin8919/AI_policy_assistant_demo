#!/bin/bash
# Hybrid AP Policy Co-Pilot System Launcher
# Seamless integration of Citation-First and SOTA architectures

echo "🚀 AP Policy Co-Pilot - Hybrid System v3.0"
echo "=========================================="
echo "Citation-First Accuracy + SOTA Intelligence"

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

# Check services
echo "🔧 Checking required services..."

# Start Weaviate if not running
if ! curl -s http://localhost:8080/v1/.well-known/ready > /dev/null; then
    echo "🚀 Starting Weaviate..."
    docker-compose up -d weaviate
    echo "⏳ Waiting for Weaviate to start..."
    sleep 15
else
    echo "✅ Weaviate is already running"
fi

# Start Neo4j if not running
if ! curl -s http://localhost:7474 > /dev/null; then
    echo "🚀 Starting Neo4j..."
    docker-compose up -d neo4j
    echo "⏳ Waiting for Neo4j to start..."
    sleep 10
else
    echo "✅ Neo4j is already running"
fi

# Check bridge tables
if [ -d "data/bridge_tables" ] && [ "$(ls -A data/bridge_tables/*.json 2>/dev/null)" ]; then
    echo "✅ Bridge tables found"
else
    echo "⚠️  Bridge tables not found - exploratory mode may be limited"
fi

echo ""
echo "🎯 Hybrid System Startup Options:"
echo ""
echo "1. 🌐 API Server only (recommended for development):"
echo "   python hybrid_orchestrator.py --action api"
echo ""
echo "2. 📊 Dashboard only (requires API server running separately):"  
echo "   streamlit run hybrid_dashboard.py"
echo ""
echo "3. 🔧 System diagnostics:"
echo "   python hybrid_orchestrator.py --action check"
echo ""
echo "4. 🔍 Command-line query:"
echo "   python hybrid_orchestrator.py --action query --query 'your query' --mode [auto|citation_first|exploratory|balanced]"
echo ""
echo "5. 🚀 Start both API + Dashboard (two terminals):"
echo "   Terminal 1: python hybrid_orchestrator.py --action api"
echo "   Terminal 2: streamlit run hybrid_dashboard.py"
echo ""

read -p "Enter choice (1-5) or press Enter for API server: " choice

case $choice in
    1|"")
        echo "🌐 Starting Hybrid API server..."
        echo "   This includes all modes: citation_first, exploratory, balanced, auto"
        python hybrid_orchestrator.py --action api
        ;;
    2)
        echo "📊 Starting Hybrid dashboard..."
        echo "   Make sure API server is running in another terminal!"
        streamlit run hybrid_dashboard.py --server.port 8501
        ;;
    3)
        echo "🔧 Running comprehensive system diagnostics..."
        python hybrid_orchestrator.py --action check
        ;;
    4)
        read -p "Enter your search query: " query
        echo ""
        echo "Available modes:"
        echo "  - auto: Intelligent mode selection (default)"
        echo "  - citation_first: Zero hallucination guarantee"
        echo "  - exploratory: SOTA bridge table insights"
        echo "  - balanced: Best of both worlds"
        echo ""
        read -p "Enter mode (or press Enter for auto): " mode
        mode=${mode:-auto}
        echo "🔍 Searching with mode: $mode"
        python hybrid_orchestrator.py --action query --query "$query" --mode "$mode"
        ;;
    5)
        echo "🚀 Starting both API server and dashboard..."
        echo "   API server will start first, then dashboard will open"
        echo "   Use Ctrl+C to stop both services"
        
        # Start API in background
        python hybrid_orchestrator.py --action api &
        API_PID=$!
        
        # Wait for API to start
        echo "⏳ Waiting for API server to start..."
        sleep 10
        
        # Start dashboard
        echo "📊 Starting dashboard..."
        streamlit run hybrid_dashboard.py --server.port 8501 &
        DASHBOARD_PID=$!
        
        # Wait for user to stop
        echo ""
        echo "✅ Both services are running!"
        echo "   API Server: http://localhost:8000"
        echo "   Dashboard: http://localhost:8501"
        echo ""
        echo "Press Ctrl+C to stop both services..."
        
        # Trap Ctrl+C to cleanup both processes
        trap "echo '🛑 Stopping services...'; kill $API_PID $DASHBOARD_PID 2>/dev/null; exit" INT
        
        # Wait for processes
        wait $API_PID $DASHBOARD_PID
        ;;
    *)
        echo "❌ Invalid choice. Starting API server by default..."
        python hybrid_orchestrator.py --action api
        ;;
esac

echo ""
echo "📋 Quick Access URLs:"
echo "   Hybrid API: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Hybrid Dashboard: http://localhost:8501"
echo "   Weaviate Console: http://localhost:8080"
echo "   Neo4j Browser: http://localhost:7474"
echo ""
echo "🎯 Hybrid Modes Available:"
echo "   - citation_first: Official/legal queries (zero hallucination)"
echo "   - exploratory: Research/trends (bridge tables + advanced RAG)"
echo "   - balanced: Comprehensive analysis (SOTA + validation)"
echo "   - auto: Intelligent mode selection based on query type"