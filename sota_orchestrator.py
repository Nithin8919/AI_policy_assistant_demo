#!/usr/bin/env python3
"""
SOTA AP Policy Co-Pilot Orchestrator
Unified orchestrator for state-of-the-art RAG system with Bridge Tables and Knowledge Graph
"""
import os
import sys
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import argparse

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "backend"))
sys.path.append(str(Path(__file__).parent / "pipeline" / "utils"))

from backend.retriever import WeaviateRetriever
from backend.graph_manager import GraphManager
from backend.gemini_rag_service import GeminiRAGService
from pipeline.utils.advanced_rag_system import AdvancedMultiAgentGraphRAG
from pipeline.utils.bridge_table_manager import BridgeTableManager

logger = logging.getLogger(__name__)

class SOTAOrchestrator:
    """State-of-the-art orchestrator combining all best components"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize SOTA system"""
        self.config = self._load_config(config_path)
        self.components = {}
        self.system_ready = False
        
        # Setup logging
        self._setup_logging()
        
        logger.info("🚀 Initializing SOTA AP Policy Co-Pilot")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            "system": {
                "name": "AP Policy Co-Pilot SOTA",
                "version": "2.0.0",
                "mode": "production"
            },
            "weaviate": {
                "url": "http://localhost:8080",
                "timeout": 30
            },
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "password"
            },
            "gemini": {
                "model": "gemini-2.5-flash",
                "temperature": 0.1,
                "max_tokens": 4000
            },
            "rag": {
                "embedding_model": "all-MiniLM-L6-v2",
                "max_results": 10,
                "confidence_threshold": 0.6,
                "enable_reranking": True,
                "enable_bridge_tables": True,
                "enable_graph_reasoning": True
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": False
            },
            "dashboard": {
                "host": "0.0.0.0", 
                "port": 8501
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.INFO
        if self.config.get("system", {}).get("mode") == "debug":
            log_level = logging.DEBUG
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('sota_system.log')
            ]
        )
    
    async def initialize_components(self):
        """Initialize all system components"""
        logger.info("🔧 Initializing system components...")
        
        try:
            # 1. Initialize Weaviate Retriever
            logger.info("   📊 Initializing Weaviate Retriever...")
            self.components['retriever'] = WeaviateRetriever()
            
            # 2. Initialize Graph Manager
            logger.info("   🕸️  Initializing Neo4j Graph Manager...")
            self.components['graph_manager'] = GraphManager(
                uri=self.config['neo4j']['uri'],
                user=self.config['neo4j']['user'],
                password=self.config['neo4j']['password']
            )
            
            # 3. Initialize Bridge Table Manager
            logger.info("   🌉 Initializing Bridge Table Manager...")
            self.components['bridge_manager'] = BridgeTableManager()
            
            # 4. Initialize Advanced RAG System
            logger.info("   🧠 Initializing Advanced Multi-Agent GraphRAG...")
            self.components['advanced_rag'] = AdvancedMultiAgentGraphRAG()
            
            # 5. Initialize Gemini Service (if API key available)
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if gemini_api_key:
                logger.info("   ⭐ Initializing Gemini RAG Service...")
                self.components['gemini_service'] = GeminiRAGService(gemini_api_key)
            else:
                logger.warning("   ⚠️  GEMINI_API_KEY not found, Gemini service disabled")
            
            # 6. Test connections
            await self._test_connections()
            
            self.system_ready = True
            logger.info("✅ All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    async def _test_connections(self):
        """Test all database connections"""
        logger.info("🧪 Testing system connections...")
        
        # Test Weaviate
        try:
            await asyncio.to_thread(self.components['retriever'].search, "test", limit=1)
            logger.info("   ✅ Weaviate connection: OK")
        except Exception as e:
            logger.error(f"   ❌ Weaviate connection failed: {e}")
            raise
        
        # Test Neo4j
        try:
            self.components['graph_manager'].test_connection()
            logger.info("   ✅ Neo4j connection: OK")
        except Exception as e:
            logger.error(f"   ❌ Neo4j connection failed: {e}")
            raise
    
    async def search(
        self, 
        query: str, 
        method: str = "auto",
        limit: int = 10,
        include_reasoning: bool = True
    ) -> Dict[str, Any]:
        """
        Unified search interface
        
        Methods:
        - auto: Intelligent method selection
        - vector: Pure vector search
        - hybrid: Vector + keyword search  
        - graph: Graph-based reasoning
        - bridge: Bridge table connections
        - advanced: Multi-agent GraphRAG
        - gemini: Gemini-powered responses
        """
        if not self.system_ready:
            raise RuntimeError("System not initialized. Call initialize_components() first.")
        
        logger.info(f"🔍 Processing query: '{query}' using method: {method}")
        
        start_time = datetime.now()
        
        try:
            if method == "auto":
                # Intelligent method selection based on query type
                method = await self._select_optimal_method(query)
                logger.info(f"   🎯 Auto-selected method: {method}")
            
            if method == "vector":
                results = await self._vector_search(query, limit)
            elif method == "hybrid":
                results = await self._hybrid_search(query, limit)
            elif method == "graph":
                results = await self._graph_search(query, limit)
            elif method == "bridge":
                results = await self._bridge_search(query, limit)
            elif method == "advanced":
                results = await self._advanced_search(query, limit, include_reasoning)
            elif method == "gemini":
                results = await self._gemini_search(query, limit)
            else:
                raise ValueError(f"Unknown search method: {method}")
            
            # Add metadata
            execution_time = (datetime.now() - start_time).total_seconds()
            results['metadata'] = {
                'method_used': method,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'system_version': self.config['system']['version']
            }
            
            logger.info(f"   ✅ Query completed in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"   ❌ Query failed: {e}")
            raise
    
    async def _select_optimal_method(self, query: str) -> str:
        """Intelligently select the best search method for the query"""
        query_lower = query.lower()
        
        # Check for specific indicators
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return "bridge"
        elif any(word in query_lower for word in ['why', 'how', 'explain', 'analyze']):
            return "gemini" if 'gemini_service' in self.components else "advanced"
        elif any(word in query_lower for word in ['trend', 'over time', 'years', 'timeline']):
            return "graph"
        elif len(query.split()) > 10:  # Complex queries
            return "advanced"
        else:
            return "hybrid"  # Default to hybrid for most queries
    
    async def _vector_search(self, query: str, limit: int) -> Dict[str, Any]:
        """Pure vector search"""
        results = await asyncio.to_thread(
            self.components['retriever'].search, 
            query, 
            limit=limit, 
            alpha=1.0  # Pure vector
        )
        return {
            'method': 'vector',
            'results': results,
            'total_results': len(results)
        }
    
    async def _hybrid_search(self, query: str, limit: int) -> Dict[str, Any]:
        """Hybrid vector + keyword search"""
        results = await asyncio.to_thread(
            self.components['retriever'].search,
            query,
            limit=limit,
            alpha=0.7  # 70% vector, 30% keyword
        )
        return {
            'method': 'hybrid',
            'results': results,
            'total_results': len(results)
        }
    
    async def _graph_search(self, query: str, limit: int) -> Dict[str, Any]:
        """Graph-based search using Neo4j"""
        graph_results = await asyncio.to_thread(
            self.components['graph_manager'].semantic_search,
            query,
            limit=limit
        )
        return {
            'method': 'graph',
            'results': graph_results,
            'total_results': len(graph_results)
        }
    
    async def _bridge_search(self, query: str, limit: int) -> Dict[str, Any]:
        """Bridge table enhanced search"""
        # First get base results
        base_results = await self._hybrid_search(query, limit)
        
        # Enhance with bridge connections
        enhanced_results = await asyncio.to_thread(
            self.components['bridge_manager'].enhance_search_results,
            base_results['results'],
            query
        )
        
        return {
            'method': 'bridge',
            'results': enhanced_results,
            'total_results': len(enhanced_results),
            'bridge_connections': True
        }
    
    async def _advanced_search(self, query: str, limit: int, include_reasoning: bool) -> Dict[str, Any]:
        """Advanced multi-agent GraphRAG search"""
        results = await self.components['advanced_rag'].process_query(
            query,
            max_results=limit,
            include_reasoning=include_reasoning
        )
        return {
            'method': 'advanced',
            'results': results.get('results', []),
            'reasoning': results.get('reasoning', []) if include_reasoning else None,
            'confidence': results.get('confidence', 0.0),
            'total_results': len(results.get('results', []))
        }
    
    async def _gemini_search(self, query: str, limit: int) -> Dict[str, Any]:
        """Gemini-powered search with citations"""
        if 'gemini_service' not in self.components:
            raise RuntimeError("Gemini service not available. Set GEMINI_API_KEY.")
        
        response = await self.components['gemini_service'].generate_response(
            query,
            max_sources=limit
        )
        
        return {
            'method': 'gemini',
            'answer': response.answer,
            'citations': [citation.__dict__ for citation in response.citations],
            'confidence': response.confidence_score,
            'retrieval_stats': response.retrieval_stats,
            'total_results': len(response.citations)
        }
    
    def start_api_server(self):
        """Start FastAPI server"""
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        
        app = FastAPI(
            title="AP Policy Co-Pilot SOTA API",
            description="State-of-the-art RAG system for AP education policy intelligence",
            version=self.config['system']['version']
        )
        
        class SearchRequest(BaseModel):
            query: str
            method: str = "auto"
            limit: int = 10
            include_reasoning: bool = True
        
        @app.post("/search")
        async def search_endpoint(request: SearchRequest):
            try:
                results = await self.search(
                    query=request.query,
                    method=request.method,
                    limit=request.limit,
                    include_reasoning=request.include_reasoning
                )
                return results
            except Exception as e:
                logger.error(f"API search failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.system_ready else "initializing",
                "components": list(self.components.keys()),
                "version": self.config['system']['version'],
                "timestamp": datetime.now().isoformat()
            }
        
        @app.get("/methods")
        async def available_methods():
            methods = ["auto", "vector", "hybrid", "graph", "bridge", "advanced"]
            if 'gemini_service' in self.components:
                methods.append("gemini")
            return {"methods": methods}
        
        logger.info(f"🌐 Starting API server on {self.config['api']['host']}:{self.config['api']['port']}")
        
        uvicorn.run(
            app,
            host=self.config['api']['host'],
            port=self.config['api']['port'],
            reload=self.config['api']['reload']
        )
    
    def start_dashboard(self):
        """Start Streamlit dashboard"""
        import subprocess
        
        dashboard_path = Path(__file__).parent / "gemini_dashboard.py"
        
        logger.info(f"📊 Starting dashboard on {self.config['dashboard']['host']}:{self.config['dashboard']['port']}")
        
        cmd = [
            "streamlit", "run", str(dashboard_path),
            "--server.port", str(self.config['dashboard']['port']),
            "--server.address", self.config['dashboard']['host']
        ]
        
        subprocess.run(cmd)
    
    async def run_system_check(self):
        """Run comprehensive system check"""
        logger.info("🔧 Running system diagnostics...")
        
        checks = {
            "weaviate_connection": False,
            "neo4j_connection": False,
            "bridge_tables_loaded": False,
            "gemini_available": False,
            "test_search": False
        }
        
        try:
            # Test connections
            await self._test_connections()
            checks["weaviate_connection"] = True
            checks["neo4j_connection"] = True
            
            # Check bridge tables
            bridge_files = list(Path("data/bridge_tables").glob("*.json"))
            checks["bridge_tables_loaded"] = len(bridge_files) > 0
            
            # Check Gemini
            checks["gemini_available"] = 'gemini_service' in self.components
            
            # Test search
            test_result = await self.search("test query", method="hybrid", limit=1)
            checks["test_search"] = len(test_result.get('results', [])) >= 0
            
        except Exception as e:
            logger.error(f"System check failed: {e}")
        
        # Report results
        logger.info("📋 System Check Results:")
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {check}: {'PASS' if status else 'FAIL'}")
        
        return checks

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='SOTA AP Policy Co-Pilot')
    parser.add_argument('--action', choices=['api', 'dashboard', 'check', 'search'], 
                       default='api', help='Action to perform')
    parser.add_argument('--query', help='Search query (for search action)')
    parser.add_argument('--method', default='auto', help='Search method')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = SOTAOrchestrator(config_path=args.config)
    await orchestrator.initialize_components()
    
    if args.action == 'api':
        orchestrator.start_api_server()
    elif args.action == 'dashboard':
        orchestrator.start_dashboard()
    elif args.action == 'check':
        await orchestrator.run_system_check()
    elif args.action == 'search':
        if not args.query:
            print("❌ Query required for search action")
            return
        
        result = await orchestrator.search(args.query, method=args.method)
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())