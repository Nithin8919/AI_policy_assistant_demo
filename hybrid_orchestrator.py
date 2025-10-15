#!/usr/bin/env python3
"""
Hybrid AP Policy Co-Pilot Orchestrator
Seamless integration of Citation-First and SOTA Bridge Table architectures
"""
import os
import sys
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime
from enum import Enum
import argparse

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "backend"))
sys.path.append(str(Path(__file__).parent / "pipeline" / "utils"))

# Citation-First imports
from backend.policy_orchestrator import PolicyOrchestrator, PolicyResponse, Citation, DocumentType
from backend.unified_retriver import UnifiedRetriever, RetrievalResult
from backend.citation_service import CitationValidator, ValidationResult
from backend.legal_processor import LegalDocumentProcessor
from backend.data_processor import StructuredDataProcessor
from backend.response_generator import CitationAwareGenerator

# SOTA imports
from pipeline.utils.advanced_rag_system import AdvancedMultiAgentGraphRAG
from pipeline.utils.bridge_table_manager import BridgeTableManager
from pipeline.utils.entity_resolver import EntityResolver
from backend.retriever import WeaviateRetriever
from backend.graph_manager import GraphManager
from backend.gemini_rag_service import GeminiRAGService

logger = logging.getLogger(__name__)

class QueryMode(Enum):
    """Query processing modes"""
    CITATION_FIRST = "citation_first"  # Official/legal use - zero hallucination
    EXPLORATORY = "exploratory"       # Research use - bridge tables + advanced RAG
    BALANCED = "balanced"             # Hybrid - SOTA retrieval + citation validation
    AUTO = "auto"                     # Intelligent mode selection

class HybridOrchestrator:
    """
    Seamless integration of Citation-First and SOTA architectures.
    
    Modes:
    - citation_first: Zero hallucination for official policy queries
    - exploratory: Cross-dataset insights with bridge tables
    - balanced: Best of both with validation
    - auto: Intelligent mode selection based on query type
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.components = {}
        self.system_ready = False
        
        # Setup logging
        self._setup_logging()
        
        logger.info("🚀 Initializing Hybrid AP Policy Co-Pilot")
        logger.info(f"   Default mode: {self.config.get('default_mode', 'balanced')}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load hybrid configuration"""
        default_config = {
            "system": {
                "name": "AP Policy Co-Pilot Hybrid",
                "version": "3.0.0",
                "default_mode": "balanced"
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
            "citation_first": {
                "min_citations": 1,
                "similarity_threshold": 0.6,
                "require_legal_chain": True,
                "temperature": 0.0
            },
            "sota_bridge": {
                "enable_bridge_tables": True,
                "enable_advanced_rag": True,
                "confidence_threshold": 0.6,
                "temperature": 0.1
            },
            "hybrid_rules": {
                "auto_mode_keywords": {
                    "citation_first": ["legal", "act", "rule", "go", "official", "policy", "law"],
                    "exploratory": ["compare", "trend", "analysis", "research", "insight"],
                    "balanced": ["statistics", "data", "enrollment", "infrastructure"]
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('hybrid_system.log')
            ]
        )
    
    async def initialize_components(self):
        """Initialize all system components"""
        logger.info("🔧 Initializing hybrid system components...")
        
        try:
            # 1. Core retrieval components
            logger.info("   📊 Initializing core retrieval...")
            self.components['weaviate_retriever'] = WeaviateRetriever()
            self.components['graph_manager'] = GraphManager(
                uri=self.config['neo4j']['uri'],
                user=self.config['neo4j']['user'],
                password=self.config['neo4j']['password']
            )
            
            # 2. Citation-First components
            logger.info("   📋 Initializing Citation-First system...")
            self.components['policy_orchestrator'] = PolicyOrchestrator()
            self.components['unified_retriever'] = UnifiedRetriever(
                weaviate_config=self.config['weaviate'],
                neo4j_config=self.config['neo4j']
            )
            self.components['citation_validator'] = CitationValidator(
                similarity_threshold=self.config['citation_first']['similarity_threshold']
            )
            self.components['legal_processor'] = LegalDocumentProcessor()
            self.components['data_processor'] = StructuredDataProcessor()
            self.components['response_generator'] = CitationAwareGenerator(
                llm_config=self.config.get('llm', {'temperature': 0.0}),
                enforce_citations=True
            )
            
            # 3. SOTA Bridge components
            logger.info("   🌉 Initializing SOTA Bridge system...")
            self.components['bridge_manager'] = BridgeTableManager()
            self.components['advanced_rag'] = AdvancedMultiAgentGraphRAG()
            self.components['entity_resolver'] = EntityResolver()
            
            # 4. Gemini service (if available)
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if gemini_api_key:
                logger.info("   ⭐ Initializing Gemini service...")
                self.components['gemini_service'] = GeminiRAGService(gemini_api_key)
            else:
                logger.warning("   ⚠️  GEMINI_API_KEY not found, Gemini features disabled")
            
            # 5. Test connections
            await self._test_connections()
            
            self.system_ready = True
            logger.info("✅ Hybrid system initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Hybrid system initialization failed: {e}")
            raise
    
    async def _test_connections(self):
        """Test all system connections"""
        logger.info("🧪 Testing hybrid system connections...")
        
        # Test Weaviate
        try:
            await asyncio.to_thread(
                self.components['weaviate_retriever'].search, 
                "test", limit=1
            )
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
        
        # Test bridge tables
        try:
            bridge_files = list(Path("data/bridge_tables").glob("*.json"))
            if bridge_files:
                logger.info(f"   ✅ Bridge tables: {len(bridge_files)} files found")
            else:
                logger.warning("   ⚠️  Bridge tables: No files found")
        except Exception as e:
            logger.warning(f"   ⚠️  Bridge tables check failed: {e}")
    
    def _select_mode(self, query: str, requested_mode: str = "auto") -> QueryMode:
        """Intelligently select processing mode"""
        if requested_mode != "auto":
            return QueryMode(requested_mode)
        
        query_lower = query.lower()
        auto_keywords = self.config['hybrid_rules']['auto_mode_keywords']
        
        # Check for citation-first keywords
        if any(keyword in query_lower for keyword in auto_keywords['citation_first']):
            return QueryMode.CITATION_FIRST
        
        # Check for exploratory keywords
        elif any(keyword in query_lower for keyword in auto_keywords['exploratory']):
            return QueryMode.EXPLORATORY
        
        # Default to balanced
        else:
            return QueryMode.BALANCED
    
    async def query(
        self,
        query: str,
        mode: str = "auto",
        limit: int = 10,
        include_reasoning: bool = True,
        require_citations: bool = None
    ) -> Dict[str, Any]:
        """
        Unified query interface supporting all modes
        
        Args:
            query: User query
            mode: Processing mode (citation_first, exploratory, balanced, auto)
            limit: Maximum results
            include_reasoning: Include reasoning chain
            require_citations: Force citation validation (auto-set based on mode)
        """
        if not self.system_ready:
            raise RuntimeError("System not initialized. Call initialize_components() first.")
        
        # Select processing mode
        selected_mode = self._select_mode(query, mode)
        
        # Auto-set citation requirements based on mode
        if require_citations is None:
            require_citations = selected_mode in [QueryMode.CITATION_FIRST, QueryMode.BALANCED]
        
        logger.info(f"🔍 Processing query: '{query}'")
        logger.info(f"   Selected mode: {selected_mode.value}")
        logger.info(f"   Citation validation: {require_citations}")
        
        start_time = datetime.now()
        
        try:
            if selected_mode == QueryMode.CITATION_FIRST:
                result = await self._citation_first_query(query, limit, include_reasoning)
            elif selected_mode == QueryMode.EXPLORATORY:
                result = await self._exploratory_query(query, limit, include_reasoning)
            elif selected_mode == QueryMode.BALANCED:
                result = await self._balanced_query(query, limit, include_reasoning)
            else:
                raise ValueError(f"Unknown mode: {selected_mode}")
            
            # Add metadata
            execution_time = (datetime.now() - start_time).total_seconds()
            result['metadata'] = {
                'mode_used': selected_mode.value,
                'mode_requested': mode,
                'execution_time': execution_time,
                'citation_validation': require_citations,
                'timestamp': datetime.now().isoformat(),
                'system_version': self.config['system']['version']
            }
            
            logger.info(f"   ✅ Query completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"   ❌ Query failed: {e}")
            raise
    
    async def _citation_first_query(self, query: str, limit: int, include_reasoning: bool) -> Dict[str, Any]:
        """Citation-first processing for official/legal queries"""
        logger.info("   📋 Using Citation-First processing...")
        
        # Use the policy orchestrator for citation-first queries
        response = await self.components['policy_orchestrator'].process_query(
            query=query,
            max_results=limit,
            require_citations=True,
            validate_legal_chain=True
        )
        
        return {
            'method': 'citation_first',
            'answer': response.answer,
            'citations': [citation.__dict__ for citation in response.citations],
            'legal_chain': response.legal_chain,
            'data_points': response.data_points,
            'confidence': response.confidence_score,
            'warnings': response.warnings,
            'total_results': len(response.citations),
            'processing_time': response.processing_time
        }
    
    async def _exploratory_query(self, query: str, limit: int, include_reasoning: bool) -> Dict[str, Any]:
        """SOTA bridge table processing for research queries"""
        logger.info("   🌉 Using SOTA Bridge processing...")
        
        # Use advanced RAG with bridge tables
        bridge_results = await self.components['advanced_rag'].process_query(
            query,
            max_results=limit,
            include_reasoning=include_reasoning
        )
        
        # Enhance with bridge connections
        enhanced_results = await asyncio.to_thread(
            self.components['bridge_manager'].enhance_search_results,
            bridge_results.get('results', []),
            query
        )
        
        return {
            'method': 'exploratory',
            'results': enhanced_results,
            'reasoning': bridge_results.get('reasoning', []) if include_reasoning else None,
            'confidence': bridge_results.get('confidence', 0.0),
            'bridge_connections': True,
            'total_results': len(enhanced_results)
        }
    
    async def _balanced_query(self, query: str, limit: int, include_reasoning: bool) -> Dict[str, Any]:
        """Balanced processing - SOTA retrieval with citation validation"""
        logger.info("   ⚖️  Using Balanced processing...")
        
        # Step 1: Get SOTA results with bridge tables
        sota_results = await self._exploratory_query(query, limit, include_reasoning)
        
        # Step 2: Get citation-first results
        citation_results = await self._citation_first_query(query, limit // 2, False)
        
        # Step 3: Merge and validate
        merged_results = self._merge_results(sota_results, citation_results)
        
        # Step 4: Validate with citation validator
        if citation_results.get('answer'):
            validation = await self.components['citation_validator'].validate(
                response_text=citation_results['answer'],
                citations=citation_results.get('citations', []),
                legal_chain=citation_results.get('legal_chain', [])
            )
            
            merged_results['validation'] = validation
            merged_results['validation_passed'] = validation.get('valid', False)
        
        return merged_results
    
    def _merge_results(self, sota_results: Dict[str, Any], citation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge SOTA and citation-first results"""
        return {
            'method': 'balanced',
            'answer': citation_results.get('answer', ''),
            'citations': citation_results.get('citations', []),
            'sota_results': sota_results.get('results', []),
            'bridge_connections': sota_results.get('bridge_connections', False),
            'reasoning': sota_results.get('reasoning', []),
            'confidence': max(
                sota_results.get('confidence', 0.0),
                citation_results.get('confidence', 0.0)
            ),
            'total_results': len(citation_results.get('citations', [])) + len(sota_results.get('results', []))
        }
    
    def start_api_server(self):
        """Start FastAPI server with hybrid endpoints"""
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        
        app = FastAPI(
            title="AP Policy Co-Pilot Hybrid API",
            description="Seamless integration of Citation-First and SOTA Bridge architectures",
            version=self.config['system']['version']
        )
        
        class QueryRequest(BaseModel):
            query: str
            mode: str = "auto"
            limit: int = 10
            include_reasoning: bool = True
            require_citations: Optional[bool] = None
        
        @app.post("/query")
        async def query_endpoint(request: QueryRequest):
            try:
                result = await self.query(
                    query=request.query,
                    mode=request.mode,
                    limit=request.limit,
                    include_reasoning=request.include_reasoning,
                    require_citations=request.require_citations
                )
                return result
            except Exception as e:
                logger.error(f"API query failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.system_ready else "initializing",
                "components": list(self.components.keys()),
                "modes": [mode.value for mode in QueryMode],
                "version": self.config['system']['version']
            }
        
        @app.get("/modes")
        async def available_modes():
            return {
                "modes": [mode.value for mode in QueryMode],
                "default_mode": self.config['system']['default_mode'],
                "descriptions": {
                    "citation_first": "Zero hallucination for official/legal queries",
                    "exploratory": "Cross-dataset insights with bridge tables",
                    "balanced": "SOTA retrieval with citation validation",
                    "auto": "Intelligent mode selection based on query"
                }
            }
        
        logger.info(f"🌐 Starting Hybrid API server on localhost:8000")
        
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    
    async def run_system_check(self):
        """Comprehensive system diagnostics"""
        logger.info("🔧 Running hybrid system diagnostics...")
        
        checks = {
            "weaviate_connection": False,
            "neo4j_connection": False,
            "bridge_tables_loaded": False,
            "citation_validator_ready": False,
            "legal_processor_ready": False,
            "advanced_rag_ready": False,
            "gemini_available": False
        }
        
        try:
            # Test connections
            await self._test_connections()
            checks["weaviate_connection"] = True
            checks["neo4j_connection"] = True
            
            # Check components
            checks["bridge_tables_loaded"] = len(list(Path("data/bridge_tables").glob("*.json"))) > 0
            checks["citation_validator_ready"] = 'citation_validator' in self.components
            checks["legal_processor_ready"] = 'legal_processor' in self.components
            checks["advanced_rag_ready"] = 'advanced_rag' in self.components
            checks["gemini_available"] = 'gemini_service' in self.components
            
            # Test each mode
            test_queries = {
                "citation_first": "What does GO 123 state about teacher recruitment?",
                "exploratory": "Compare enrollment trends across districts",
                "balanced": "What are the infrastructure statistics for 2023?"
            }
            
            mode_results = {}
            for mode, test_query in test_queries.items():
                try:
                    result = await self.query(test_query, mode=mode, limit=3)
                    mode_results[mode] = "✅ PASS"
                except Exception as e:
                    mode_results[mode] = f"❌ FAIL: {str(e)[:50]}"
            
        except Exception as e:
            logger.error(f"System check failed: {e}")
        
        # Report results
        logger.info("📋 Hybrid System Check Results:")
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {check}: {'PASS' if status else 'FAIL'}")
        
        logger.info("📋 Mode Test Results:")
        for mode, result in mode_results.items():
            logger.info(f"   {result} {mode}")
        
        return checks, mode_results

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Hybrid AP Policy Co-Pilot')
    parser.add_argument('--action', choices=['api', 'check', 'query'], 
                       default='api', help='Action to perform')
    parser.add_argument('--query', help='Search query (for query action)')
    parser.add_argument('--mode', default='auto', 
                       choices=['auto', 'citation_first', 'exploratory', 'balanced'],
                       help='Query processing mode')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize hybrid orchestrator
    orchestrator = HybridOrchestrator(config_path=args.config)
    await orchestrator.initialize_components()
    
    if args.action == 'api':
        orchestrator.start_api_server()
    elif args.action == 'check':
        await orchestrator.run_system_check()
    elif args.action == 'query':
        if not args.query:
            print("❌ Query required for query action")
            return
        
        result = await orchestrator.query(args.query, mode=args.mode)
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())