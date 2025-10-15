#!/usr/bin/env python3
"""
AP Policy Co-Pilot - Citation-First Policy Reasoning System
Built for verifiable, non-hallucinating policy intelligence
"""
import os
import sys
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import asyncio

from dataclasses import dataclass, field
from enum import Enum

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "backend"))
sys.path.append(str(Path(__file__).parent / "pipeline" / "utils"))

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Legal document hierarchy"""
    CONSTITUTION = "constitution"
    CENTRAL_ACT = "central_act"
    STATE_ACT = "state_act"
    CENTRAL_RULE = "central_rule"
    STATE_RULE = "state_rule"
    GOVERNMENT_ORDER = "go"
    CIRCULAR = "circular"
    FRAMEWORK = "framework"
    STATISTICS = "statistics"
    BUDGET = "budget"
    AUDIT = "audit"

@dataclass
class Citation:
    """Verifiable citation with full provenance"""
    doc_id: str
    doc_title: str
    doc_type: DocumentType
    section: Optional[str] = None
    clause: Optional[str] = None
    page_number: Optional[int] = None
    issued_date: Optional[str] = None
    effective_date: Optional[str] = None
    supersedes: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    source_url: Optional[str] = None
    excerpt: str = ""
    confidence: float = 1.0
    
    def to_citation_string(self) -> str:
        """Format as human-readable citation"""
        parts = [f"{self.doc_title}"]
        if self.section:
            parts.append(f"Section {self.section}")
        if self.clause:
            parts.append(f"Clause {self.clause}")
        if self.issued_date:
            parts.append(f"({self.issued_date})")
        if self.page_number:
            parts.append(f"Page {self.page_number}")
        return ", ".join(parts)

@dataclass
class PolicyResponse:
    """Policy reasoning response with mandatory citations"""
    query: str
    answer: str
    citations: List[Citation]
    legal_chain: List[str]  # Act → Rule → GO hierarchy
    data_points: List[Dict[str, Any]]  # Statistics with sources
    confidence_score: float
    retrieval_method: str
    processing_time: float
    warnings: List[str] = field(default_factory=list)
    
    def validate_citations(self) -> bool:
        """Ensure every claim has a citation"""
        if not self.citations:
            self.warnings.append("No citations provided - response may be unreliable")
            return False
        return True

class PolicyOrchestrator:
    """
    Citation-first policy reasoning orchestrator.
    Enforces: retrieve → validate → cite → generate pattern.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.components = {}
        self.system_ready = False
        self._setup_logging()
        
        logger.info("🏛️ Initializing AP Policy Co-Pilot (Citation-First Mode)")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            "system": {
                "name": "AP Policy Co-Pilot",
                "version": "3.0.0",
                "mode": "citation_first",  # Enforces citation validation
                "allow_generative": False  # No free-form generation
            },
            "data_sources": {
                "legal_docs": "data/legal_frameworks",
                "statistics": "data/statistics",
                "budget_audit": "data/budget_audit",
                "corpus_index": "data/corpus_index.csv"
            },
            "databases": {
                "weaviate": {
                    "url": "http://localhost:8080",
                    "timeout": 30
                },
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "password"
                }
            },
            "retrieval": {
                "embedding_model": "all-MiniLM-L6-v2",
                "max_results": 15,
                "min_confidence": 0.7,
                "enable_legal_hierarchy": True,
                "enable_temporal_reasoning": True
            },
            "citation": {
                "require_citations": True,
                "min_citations_per_claim": 1,
                "validate_chain": True  # Act→Rule→GO validation
            },
            "llm": {
                "provider": "gemini",  # or "claude", "gpt4"
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.0,  # Deterministic for policy
                "max_tokens": 2000,
                "system_prompt": "You are a policy reasoning assistant. ONLY paraphrase information from retrieved documents. NEVER invent facts. Every statement must be traceable to a citation."
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._deep_update(default_config, user_config)
        
        return default_config
    
    def _deep_update(self, base: Dict, update: Dict) -> Dict:
        """Deep merge configuration dictionaries"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                base[key] = self._deep_update(base.get(key, {}), value)
            else:
                base[key] = value
        return base
    
    def _setup_logging(self):
        """Configure logging"""
        log_level = logging.DEBUG if self.config["system"]["mode"] == "debug" else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'policy_copilot_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("🔧 Initializing Policy Co-Pilot components...")
        
        try:
            # 1. Initialize Legal Document Processor
            from legal_processor import LegalDocumentProcessor
            logger.info("   ⚖️  Initializing Legal Document Processor...")
            self.components['legal_processor'] = LegalDocumentProcessor(
                corpus_index=self.config["data_sources"]["corpus_index"]
            )
            
            # 2. Initialize Data Processor (Statistics, Budget, etc.)
            from data_processor import StructuredDataProcessor
            logger.info("   📊 Initializing Structured Data Processor...")
            self.components['data_processor'] = StructuredDataProcessor()
            
            # 3. Initialize Unified Retriever (Vector + Graph + Legal Hierarchy)
            from unified_retriever import UnifiedRetriever
            logger.info("   🔍 Initializing Unified Retriever...")
            self.components['retriever'] = UnifiedRetriever(
                weaviate_config=self.config["databases"]["weaviate"],
                neo4j_config=self.config["databases"]["neo4j"],
                embedding_model=self.config["retrieval"]["embedding_model"]
            )
            
            # 4. Initialize Citation Validator
            from citation_validator import CitationValidator
            logger.info("   ✓ Initializing Citation Validator...")
            self.components['citation_validator'] = CitationValidator(
                min_citations=self.config["citation"]["min_citations_per_claim"],
                validate_chain=self.config["citation"]["validate_chain"]
            )
            
            # 5. Initialize Response Generator (LLM with citation constraints)
            from response_generator import CitationAwareGenerator
            logger.info("   🤖 Initializing Citation-Aware Response Generator...")
            self.components['generator'] = CitationAwareGenerator(
                llm_config=self.config["llm"],
                enforce_citations=self.config["system"]["allow_generative"] == False
            )
            
            # 6. Test all connections
            await self._health_check()
            
            self.system_ready = True
            logger.info("✅ Policy Co-Pilot ready for citation-first reasoning!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            raise
    
    async def _health_check(self):
        """Verify all system components are operational"""
        logger.info("🏥 Running system health checks...")
        
        checks = {
            "weaviate": self.components['retriever'].test_weaviate_connection(),
            "neo4j": self.components['retriever'].test_neo4j_connection(),
            "legal_corpus": self.components['legal_processor'].verify_corpus(),
            "data_corpus": self.components['data_processor'].verify_corpus()
        }
        
        for component, status in checks.items():
            if status:
                logger.info(f"   ✅ {component}: OK")
            else:
                logger.error(f"   ❌ {component}: FAILED")
                raise RuntimeError(f"{component} health check failed")
    
    async def query(
        self,
        query: str,
        query_type: str = "auto",  # 'legal', 'data', 'combined', 'auto'
        require_citations: bool = True,
        max_results: int = 10
    ) -> PolicyResponse:
        """
        Main query interface with citation-first architecture.
        
        Process:
        1. Analyze query intent
        2. Retrieve relevant documents (legal + data)
        3. Build legal hierarchy chain (Act→Rule→GO)
        4. Validate all citations
        5. Generate response (LLM paraphrases only)
        6. Final citation validation
        """
        if not self.system_ready:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        logger.info(f"📥 Processing query: '{query}'")
        start_time = datetime.now()
        
        try:
            # Step 1: Analyze query
            query_analysis = await self._analyze_query(query, query_type)
            logger.info(f"   🎯 Query type: {query_analysis['type']}, Intent: {query_analysis['intent']}")
            
            # Step 2: Multi-source retrieval
            retrieval_results = await self._retrieve_documents(
                query=query,
                query_analysis=query_analysis,
                max_results=max_results
            )
            
            logger.info(f"   📚 Retrieved {len(retrieval_results['legal'])} legal docs, "
                       f"{len(retrieval_results['data'])} data sources")
            
            # Step 3: Build legal chain (Act → Rule → GO)
            legal_chain = await self._build_legal_chain(retrieval_results['legal'])
            logger.info(f"   ⚖️  Legal chain: {' → '.join(legal_chain)}")
            
            # Step 4: Extract citations
            citations = await self._extract_citations(retrieval_results)
            logger.info(f"   📝 Extracted {len(citations)} citations")
            
            # Step 5: Generate response (LLM constrained to paraphrase only)
            response_text = await self._generate_response(
                query=query,
                retrieval_results=retrieval_results,
                citations=citations,
                legal_chain=legal_chain
            )
            
            # Step 6: Validate citations in response
            validation_result = await self._validate_response_citations(
                response_text=response_text,
                available_citations=citations
            )
            
            if not validation_result['valid'] and require_citations:
                raise ValueError(f"Citation validation failed: {validation_result['errors']}")
            
            # Build final response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            policy_response = PolicyResponse(
                query=query,
                answer=response_text,
                citations=citations,
                legal_chain=legal_chain,
                data_points=retrieval_results['data'],
                confidence_score=validation_result['confidence'],
                retrieval_method=query_analysis['optimal_method'],
                processing_time=processing_time,
                warnings=validation_result.get('warnings', [])
            )
            
            # Final validation
            if not policy_response.validate_citations():
                logger.warning("⚠️  Response lacks sufficient citations")
            
            logger.info(f"   ✅ Response generated in {processing_time:.2f}s with {len(citations)} citations")
            return policy_response
            
        except Exception as e:
            logger.error(f"   ❌ Query processing failed: {e}", exc_info=True)
            raise
    
    async def _analyze_query(self, query: str, query_type: str) -> Dict[str, Any]:
        """Analyze query to determine optimal retrieval strategy"""
        analysis = {
            'original_query': query,
            'type': query_type,
            'intent': None,
            'entities': [],
            'temporal_scope': None,
            'optimal_method': 'hybrid'
        }
        
        query_lower = query.lower()
        
        # Detect intent
        if any(word in query_lower for word in ['which act', 'which rule', 'which go', 'legal', 'law']):
            analysis['intent'] = 'legal_lookup'
            analysis['optimal_method'] = 'legal_hierarchy'
        elif any(word in query_lower for word in ['statistics', 'data', 'how many', 'enrollment', 'dropout']):
            analysis['intent'] = 'data_lookup'
            analysis['optimal_method'] = 'data_retrieval'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            analysis['intent'] = 'comparison'
            analysis['optimal_method'] = 'bridge_comparison'
        elif any(word in query_lower for word in ['trend', 'over time', 'change', 'growth']):
            analysis['intent'] = 'temporal_analysis'
            analysis['optimal_method'] = 'temporal_graph'
        else:
            analysis['intent'] = 'general'
            analysis['optimal_method'] = 'hybrid'
        
        # Auto-detect type if not specified
        if query_type == 'auto':
            if analysis['intent'] == 'legal_lookup':
                analysis['type'] = 'legal'
            elif analysis['intent'] == 'data_lookup':
                analysis['type'] = 'data'
            else:
                analysis['type'] = 'combined'
        
        # Extract entities (districts, years, indicators)
        # TODO: Use NER model here
        analysis['entities'] = self._extract_entities_simple(query)
        
        return analysis
    
    def _extract_entities_simple(self, query: str) -> List[str]:
        """Simple entity extraction (replace with NER model)"""
        entities = []
        
        # AP districts
        districts = ['Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Krishna', 
                    'Kurnool', 'Prakasam', 'Nellore', 'Srikakulam', 'Visakhapatnam', 
                    'Vizianagaram', 'West Godavari', 'YSR Kadapa', 'Alluri Sitharama Raju']
        
        for district in districts:
            if district.lower() in query.lower():
                entities.append(f"DISTRICT:{district}")
        
        # Years
        import re
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        entities.extend([f"YEAR:{year}" for year in years])
        
        return entities
    
    async def _retrieve_documents(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        max_results: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Multi-source document retrieval"""
        
        retrieval_results = {
            'legal': [],
            'data': [],
            'metadata': {}
        }
        
        # Retrieve based on query type
        if query_analysis['type'] in ['legal', 'combined']:
            legal_results = await self.components['retriever'].retrieve_legal(
                query=query,
                max_results=max_results,
                filters={'entities': query_analysis['entities']}
            )
            retrieval_results['legal'] = legal_results
        
        if query_analysis['type'] in ['data', 'combined']:
            data_results = await self.components['retriever'].retrieve_data(
                query=query,
                max_results=max_results,
                filters={'entities': query_analysis['entities']}
            )
            retrieval_results['data'] = data_results
        
        return retrieval_results
    
    async def _build_legal_chain(self, legal_documents: List[Dict[str, Any]]) -> List[str]:
        """Build Act → Rule → GO hierarchy chain"""
        chain = []
        doc_hierarchy = {}
        
        for doc in legal_documents:
            doc_type = doc.get('doc_type')
            if doc_type in ['act', 'rule', 'go']:
                doc_hierarchy[doc_type] = doc.get('title', 'Unknown')
        
        # Build chain in legal hierarchy order
        if 'act' in doc_hierarchy:
            chain.append(f"Act: {doc_hierarchy['act']}")
        if 'rule' in doc_hierarchy:
            chain.append(f"Rule: {doc_hierarchy['rule']}")
        if 'go' in doc_hierarchy:
            chain.append(f"GO: {doc_hierarchy['go']}")
        
        return chain
    
    async def _extract_citations(self, retrieval_results: Dict[str, Any]) -> List[Citation]:
        """Extract structured citations from retrieved documents"""
        citations = []
        
        # Extract from legal documents
        for doc in retrieval_results.get('legal', []):
            citation = Citation(
                doc_id=doc.get('id', 'unknown'),
                doc_title=doc.get('title', 'Untitled'),
                doc_type=DocumentType(doc.get('doc_type', 'go')),
                section=doc.get('section'),
                clause=doc.get('clause'),
                page_number=doc.get('page'),
                issued_date=doc.get('issued_date'),
                effective_date=doc.get('effective_date'),
                source_url=doc.get('source_url'),
                excerpt=doc.get('text', '')[:500],
                confidence=doc.get('score', 0.8)
            )
            citations.append(citation)
        
        # Extract from data documents
        for doc in retrieval_results.get('data', []):
            citation = Citation(
                doc_id=doc.get('id', 'unknown'),
                doc_title=doc.get('source', 'Unknown Source'),
                doc_type=DocumentType.STATISTICS,
                excerpt=f"{doc.get('indicator', 'N/A')}: {doc.get('value', 'N/A')} ({doc.get('year', 'N/A')})",
                confidence=doc.get('score', 0.8)
            )
            citations.append(citation)
        
        return citations
    
    async def _generate_response(
        self,
        query: str,
        retrieval_results: Dict[str, Any],
        citations: List[Citation],
        legal_chain: List[str]
    ) -> str:
        """Generate response using LLM (constrained to paraphrase only)"""
        
        # Build context from retrieved documents
        context = self._build_context(retrieval_results, citations)
        
        # Generate response using citation-aware generator
        response = await self.components['generator'].generate(
            query=query,
            context=context,
            legal_chain=legal_chain,
            citations=citations
        )
        
        return response
    
    def _build_context(self, retrieval_results: Dict[str, Any], citations: List[Citation]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        
        # Legal context
        if retrieval_results.get('legal'):
            context_parts.append("### Legal Framework:")
            for doc in retrieval_results['legal'][:5]:
                context_parts.append(f"- {doc.get('title')}: {doc.get('text', '')[:300]}")
        
        # Data context
        if retrieval_results.get('data'):
            context_parts.append("\n### Statistical Data:")
            for doc in retrieval_results['data'][:5]:
                context_parts.append(
                    f"- {doc.get('indicator', 'N/A')}: {doc.get('value', 'N/A')} "
                    f"({doc.get('district', 'N/A')}, {doc.get('year', 'N/A')})"
                )
        
        return "\n".join(context_parts)
    
    async def _validate_response_citations(
        self,
        response_text: str,
        available_citations: List[Citation]
    ) -> Dict[str, Any]:
        """Validate that response only contains cited information"""
        
        validation = await self.components['citation_validator'].validate(
            response_text=response_text,
            citations=available_citations
        )
        
        return validation
    
    async def batch_query(self, queries: List[str], **kwargs) -> List[PolicyResponse]:
        """Process multiple queries in batch"""
        tasks = [self.query(q, **kwargs) for q in queries]
        return await asyncio.gather(*tasks)
    
    def export_response(self, response: PolicyResponse, format: str = "markdown") -> str:
        """Export response in various formats"""
        if format == "markdown":
            return self._export_markdown(response)
        elif format == "json":
            return json.dumps(response.__dict__, default=str, indent=2)
        elif format == "html":
            return self._export_html(response)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_markdown(self, response: PolicyResponse) -> str:
        """Export response as markdown"""
        md = [
            f"# Query: {response.query}\n",
            f"## Answer\n{response.answer}\n",
            f"## Legal Chain\n{' → '.join(response.legal_chain)}\n",
            f"## Citations\n"
        ]
        
        for i, citation in enumerate(response.citations, 1):
            md.append(f"{i}. {citation.to_citation_string()}")
        
        md.append(f"\n## Metadata")
        md.append(f"- Confidence: {response.confidence_score:.2%}")
        md.append(f"- Processing Time: {response.processing_time:.2f}s")
        md.append(f"- Method: {response.retrieval_method}")
        
        if response.warnings:
            md.append(f"\n⚠️ **Warnings:**")
            for warning in response.warnings:
                md.append(f"- {warning}")
        
        return "\n".join(md)
    
    def _export_html(self, response: PolicyResponse) -> str:
        """Export response as HTML"""
        # TODO: Implement HTML export
        raise NotImplementedError("HTML export not yet implemented")

# CLI Interface
async def main():
    parser = argparse.ArgumentParser(description="AP Policy Co-Pilot")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--test", action="store_true", help="Run test queries")
    parser.add_argument("--export", type=str, choices=["markdown", "json", "html"], 
                       default="markdown", help="Export format")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PolicyOrchestrator(config_path=args.config)
    await orchestrator.initialize()
    
    if args.test:
        # Run test queries
        test_queries = [
            "What are the responsibilities of School Management Committees under AP law?",
            "What was the dropout rate among ST students in 2016-17?",
            "Which GO governs Nadu-Nedu infrastructure improvement?",
            "How many schools offered Telugu medium in 2016-17?",
            "Under which rule can private school fees be regulated?"
        ]
        
        logger.info("🧪 Running test queries...")
        responses = await orchestrator.batch_query(test_queries)
        
        for i, response in enumerate(responses, 1):
            print(f"\n{'='*80}")
            print(f"Test Query {i}/{len(test_queries)}")
            print('='*80)
            print(orchestrator.export_response(response, format=args.export))
    
    elif args.query:
        # Process single query
        response = await orchestrator.query(args.query)
        print(orchestrator.export_response(response, format=args.export))
    
    else:
        # Interactive mode
        print("\n🏛️  AP Policy Co-Pilot - Citation-First Policy Reasoning")
        print("=" * 60)
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                response = await orchestrator.query(query)
                print("\n" + orchestrator.export_response(response, format=args.export))
                print("\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"❌ Error: {e}\n")
        
        print("\nGoodbye! 👋")

if __name__ == "__main__":
    asyncio.run(main())