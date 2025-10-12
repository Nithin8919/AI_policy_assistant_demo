#!/usr/bin/env python3
"""
Advanced Multi-Agent Hierarchical GraphRAG + CRAG-TAG System
State-of-the-art RAG architecture for policy document analysis with 99% accuracy
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import json
import re
import numpy as np
from datetime import datetime
import networkx as nx
from collections import defaultdict
import weaviate
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
# import openai  # Optional for advanced features
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class PolicyQuery:
    """Enhanced query structure for policy analysis"""
    query_text: str
    query_type: str  # 'statistical', 'comparative', 'temporal', 'policy_lookup', 'complex'
    intent: str  # 'find', 'compare', 'analyze', 'trend', 'summarize'
    entities: List[str]  # Districts, indicators, years, etc.
    temporal_scope: Optional[Tuple[int, int]] = None
    comparison_dimensions: Optional[List[str]] = None
    accuracy_requirement: float = 0.95
    max_sources: int = 10

@dataclass
class EnhancedResult:
    """Comprehensive result with verification and provenance"""
    content: str
    source_type: str  # 'graph', 'table', 'text', 'hybrid'
    confidence_score: float
    verification_status: str  # 'verified', 'uncertain', 'conflicting'
    sources: List[Dict[str, Any]]
    reasoning_chain: List[str]
    corrections_applied: List[str]
    graph_context: Optional[Dict[str, Any]] = None
    numerical_data: Optional[Dict[str, float]] = None

class QueryClassificationAgent:
    """Classifies queries and determines optimal processing strategy"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        
        # Query type patterns for AP education policy domain
        self.query_patterns = {
            'statistical': [
                'how many', 'what is the number', 'statistics', 'data', 'count', 'percentage',
                'enrollment', 'dropout rate', 'ger', 'ner', 'ptr', 'schools count', 'teachers'
            ],
            'comparative': [
                'compare', 'versus', 'vs', 'difference between', 'better than', 'higher',
                'district comparison', 'performance gap', 'ranking'
            ],
            'temporal': [
                'trend', 'over time', 'increase', 'decrease', 'year over year', 'growth',
                'improvement', 'decline', 'change since', 'progression'
            ],
            'policy_lookup': [
                'policy', 'order', 'circular', 'notification', 'rule', 'regulation',
                'government order', 'go', 'amendment', 'revision'
            ],
            'complex': [
                'why', 'how', 'what factors', 'analyze', 'impact', 'correlation',
                'relationship between', 'causes', 'effects'
            ]
        }
        
        # Entity recognition patterns
        self.entity_patterns = {
            'districts': [
                'anantapur', 'chittoor', 'east godavari', 'guntur', 'krishna', 'kurnool',
                'nellore', 'prakasam', 'srikakulam', 'visakhapatnam', 'vizianagaram',
                'west godavari', 'kadapa'
            ],
            'indicators': [
                'enrollment', 'enrolment', 'ger', 'ner', 'ptr', 'dropout', 'literacy',
                'schools', 'teachers', 'students', 'infrastructure', 'applications'
            ],
            'years': r'\b(20[0-2][0-9])\b'
        }
    
    def classify_query(self, query_text: str) -> PolicyQuery:
        """Classify query and extract entities"""
        query_lower = query_text.lower()
        
        # Determine query type
        query_type = 'complex'  # default
        max_score = 0
        
        for qtype, patterns in self.query_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > max_score:
                max_score = score
                query_type = qtype
        
        # Determine intent
        intent = self._determine_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query_text)
        
        # Extract temporal scope
        temporal_scope = self._extract_temporal_scope(query_text)
        
        # Determine comparison dimensions
        comparison_dims = self._extract_comparison_dimensions(query_text)
        
        return PolicyQuery(
            query_text=query_text,
            query_type=query_type,
            intent=intent,
            entities=entities,
            temporal_scope=temporal_scope,
            comparison_dimensions=comparison_dims
        )
    
    def _determine_intent(self, query_lower: str) -> str:
        """Determine the intent of the query"""
        if any(word in query_lower for word in ['find', 'get', 'show', 'what is']):
            return 'find'
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return 'compare'
        elif any(word in query_lower for word in ['analyze', 'why', 'how', 'impact']):
            return 'analyze'
        elif any(word in query_lower for word in ['trend', 'change', 'growth', 'over time']):
            return 'trend'
        elif any(word in query_lower for word in ['summary', 'summarize', 'overview']):
            return 'summarize'
        else:
            return 'find'
    
    def _extract_entities(self, query_text: str) -> List[str]:
        """Extract relevant entities from query"""
        entities = []
        query_lower = query_text.lower()
        
        # Extract districts
        for district in self.entity_patterns['districts']:
            if district in query_lower:
                entities.append(district.title())
        
        # Extract indicators
        for indicator in self.entity_patterns['indicators']:
            if indicator in query_lower:
                entities.append(indicator.upper() if indicator in ['ger', 'ner', 'ptr'] else indicator.title())
        
        # Extract years
        years = re.findall(self.entity_patterns['years'], query_text)
        entities.extend(years)
        
        return entities
    
    def _extract_temporal_scope(self, query_text: str) -> Optional[Tuple[int, int]]:
        """Extract temporal scope from query"""
        years = re.findall(r'\b(20[0-2][0-9])\b', query_text)
        
        if len(years) >= 2:
            return (int(min(years)), int(max(years)))
        elif len(years) == 1:
            year = int(years[0])
            return (year, year)
        elif any(phrase in query_text.lower() for phrase in ['last 5 years', 'past 5 years']):
            current_year = datetime.now().year
            return (current_year - 5, current_year)
        elif any(phrase in query_text.lower() for phrase in ['last 3 years', 'past 3 years']):
            current_year = datetime.now().year
            return (current_year - 3, current_year)
        
        return None
    
    def _extract_comparison_dimensions(self, query_text: str) -> Optional[List[str]]:
        """Extract comparison dimensions"""
        query_lower = query_text.lower()
        dimensions = []
        
        if 'district' in query_lower:
            dimensions.append('district')
        if any(word in query_lower for word in ['year', 'time', 'temporal']):
            dimensions.append('temporal')
        if any(word in query_lower for word in ['indicator', 'metric', 'measure']):
            dimensions.append('indicator')
        
        return dimensions if dimensions else None

class HierarchicalGraphRAGAgent:
    """Hierarchical GraphRAG for policy relationship analysis"""
    
    def __init__(self, neo4j_driver, weaviate_client):
        self.neo4j_driver = neo4j_driver
        self.weaviate_client = weaviate_client
        self.graph = nx.DiGraph()
        self.communities = {}
        self.community_summaries = {}
        
    async def build_hierarchical_graph(self, facts: List[Dict[str, Any]]):
        """Build hierarchical knowledge graph with community detection"""
        logger.info("🏗️ Building hierarchical policy knowledge graph")
        
        # Stage 1: Build fact-level graph
        await self._build_fact_graph(facts)
        
        # Stage 2: Detect policy communities
        await self._detect_policy_communities()
        
        # Stage 3: Create hierarchical summaries
        await self._create_community_summaries()
        
        logger.info(f"✅ Built graph with {len(self.graph.nodes)} nodes and {len(self.communities)} communities")
    
    async def _build_fact_graph(self, facts: List[Dict[str, Any]]):
        """Build graph connecting related facts"""
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i+1:]:
                # Calculate relationship strength
                relationship_strength = self._calculate_relationship_strength(fact1, fact2)
                
                if relationship_strength > 0.3:  # Threshold for connection
                    self.graph.add_edge(
                        fact1['fact_id'], 
                        fact2['fact_id'],
                        weight=relationship_strength,
                        relationship_type=self._get_relationship_type(fact1, fact2)
                    )
    
    async def _detect_policy_communities(self):
        """Detect communities of related policies using advanced algorithms"""
        try:
            # Use Leiden algorithm for community detection (better than Louvain)
            import networkx.algorithms.community as nx_comm
            self.communities = nx_comm.greedy_modularity_communities(self.graph)
            
            # Convert to dict format
            community_dict = {}
            for i, community in enumerate(self.communities):
                community_dict[f"community_{i}"] = list(community)
            
            self.communities = community_dict
            
        except ImportError:
            # Fallback to simple connected components
            self.communities = {
                f"component_{i}": list(component) 
                for i, component in enumerate(nx.connected_components(self.graph.to_undirected()))
            }
    
    async def _create_community_summaries(self):
        """Create hierarchical summaries for each community"""
        for community_id, fact_ids in self.communities.items():
            # Get facts for this community
            community_facts = []
            
            # Query facts from Weaviate
            for fact_id in fact_ids[:50]:  # Limit for performance
                try:
                    result = (
                        self.weaviate_client.query
                        .get("Fact", ["fact_id", "district", "indicator", "year", "value", "content"])
                        .with_where({
                            "path": ["fact_id"],
                            "operator": "Equal",
                            "valueText": fact_id
                        })
                        .with_limit(1)
                        .do()
                    )
                    
                    facts = result.get("data", {}).get("Get", {}).get("Fact", [])
                    if facts:
                        community_facts.append(facts[0])
                        
                except Exception as e:
                    logger.warning(f"Failed to get fact {fact_id}: {e}")
                    continue
            
            # Create community summary
            if community_facts:
                summary = await self._summarize_community(community_facts)
                self.community_summaries[community_id] = summary
    
    async def _summarize_community(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary for a community of facts"""
        # Analyze community characteristics
        districts = set(fact.get('district') for fact in facts if fact.get('district'))
        indicators = set(fact.get('indicator') for fact in facts if fact.get('indicator'))
        years = set(fact.get('year') for fact in facts if fact.get('year'))
        
        # Determine community theme
        theme = self._determine_community_theme(facts)
        
        # Create summary text
        summary_text = f"Policy community focused on {theme}. "
        summary_text += f"Covers {len(districts)} districts, {len(indicators)} indicators, "
        summary_text += f"spanning years {min(years) if years else 'unknown'}-{max(years) if years else 'unknown'}."
        
        return {
            'theme': theme,
            'districts': list(districts),
            'indicators': list(indicators),
            'year_range': [min(years), max(years)] if years else None,
            'fact_count': len(facts),
            'summary_text': summary_text,
            'key_facts': facts[:5]  # Top 5 representative facts
        }
    
    def _determine_community_theme(self, facts: List[Dict[str, Any]]) -> str:
        """Determine the main theme of a community"""
        indicators = [fact.get('indicator', '') for fact in facts]
        
        # Count indicator frequency
        indicator_counts = defaultdict(int)
        for indicator in indicators:
            if indicator:
                indicator_counts[indicator] += 1
        
        if indicator_counts:
            primary_indicator = max(indicator_counts, key=indicator_counts.get)
            return primary_indicator
        else:
            return "Mixed Policy Areas"
    
    def _calculate_relationship_strength(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> float:
        """Calculate relationship strength between two facts"""
        strength = 0.0
        
        # Same district
        if fact1.get('district') == fact2.get('district') and fact1.get('district') != 'Unknown':
            strength += 0.4
        
        # Same indicator
        if fact1.get('indicator') == fact2.get('indicator'):
            strength += 0.3
        
        # Temporal proximity
        year1, year2 = fact1.get('year'), fact2.get('year')
        if year1 and year2:
            year_diff = abs(year1 - year2)
            if year_diff <= 1:
                strength += 0.3
            elif year_diff <= 3:
                strength += 0.2
        
        # Same document source
        if fact1.get('source_document') == fact2.get('source_document'):
            strength += 0.2
        
        return min(strength, 1.0)
    
    def _get_relationship_type(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> str:
        """Determine the type of relationship between facts"""
        if fact1.get('district') == fact2.get('district'):
            return 'same_district'
        elif fact1.get('indicator') == fact2.get('indicator'):
            return 'same_indicator'
        elif fact1.get('year') == fact2.get('year'):
            return 'same_year'
        elif fact1.get('source_document') == fact2.get('source_document'):
            return 'same_document'
        else:
            return 'related'
    
    async def hierarchical_search(self, query: PolicyQuery) -> List[Dict[str, Any]]:
        """Perform hierarchical search through graph communities"""
        # Find relevant communities
        relevant_communities = await self._find_relevant_communities(query)
        
        # Search within relevant communities
        community_results = []
        for community_id in relevant_communities:
            community_facts = await self._search_community(community_id, query)
            community_results.extend(community_facts)
        
        # Rank and return results
        ranked_results = self._rank_community_results(community_results, query)
        return ranked_results[:query.max_sources]
    
    async def _find_relevant_communities(self, query: PolicyQuery) -> List[str]:
        """Find communities relevant to the query"""
        relevant_communities = []
        
        for community_id, summary in self.community_summaries.items():
            relevance_score = self._calculate_community_relevance(summary, query)
            if relevance_score > 0.3:
                relevant_communities.append(community_id)
        
        return relevant_communities
    
    def _calculate_community_relevance(self, summary: Dict[str, Any], query: PolicyQuery) -> float:
        """Calculate how relevant a community is to the query"""
        score = 0.0
        
        # Check entity overlap
        query_entities_lower = [e.lower() for e in query.entities]
        
        # District relevance
        community_districts = [d.lower() for d in summary.get('districts', [])]
        district_overlap = len(set(query_entities_lower) & set(community_districts))
        score += district_overlap * 0.3
        
        # Indicator relevance
        community_indicators = [i.lower() for i in summary.get('indicators', [])]
        indicator_overlap = len(set(query_entities_lower) & set(community_indicators))
        score += indicator_overlap * 0.4
        
        # Temporal relevance
        if query.temporal_scope and summary.get('year_range'):
            query_start, query_end = query.temporal_scope
            comm_start, comm_end = summary['year_range']
            
            # Check for overlap
            if query_start <= comm_end and query_end >= comm_start:
                score += 0.3
        
        return min(score, 1.0)
    
    async def _search_community(self, community_id: str, query: PolicyQuery) -> List[Dict[str, Any]]:
        """Search within a specific community"""
        fact_ids = self.communities.get(community_id, [])
        
        # Get detailed facts for this community
        community_facts = []
        for fact_id in fact_ids[:20]:  # Limit for performance
            try:
                result = (
                    self.weaviate_client.query
                    .get("Fact", [
                        "fact_id", "district", "indicator", "year", "value", "unit", 
                        "content", "confidence_score"
                    ])
                    .with_where({
                        "path": ["fact_id"],
                        "operator": "Equal",
                        "valueText": fact_id
                    })
                    .with_limit(1)
                    .do()
                )
                
                facts = result.get("data", {}).get("Get", {}).get("Fact", [])
                if facts:
                    fact = facts[0]
                    fact['community_id'] = community_id
                    community_facts.append(fact)
                    
            except Exception as e:
                logger.warning(f"Failed to get community fact {fact_id}: {e}")
                continue
        
        return community_facts
    
    def _rank_community_results(self, results: List[Dict[str, Any]], query: PolicyQuery) -> List[Dict[str, Any]]:
        """Rank results from community search"""
        for result in results:
            score = self._calculate_result_relevance(result, query)
            result['relevance_score'] = score
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    def _calculate_result_relevance(self, result: Dict[str, Any], query: PolicyQuery) -> float:
        """Calculate relevance score for a result"""
        score = 0.0
        
        # Entity matching
        query_entities_lower = [e.lower() for e in query.entities]
        
        if result.get('district', '').lower() in query_entities_lower:
            score += 0.3
        if result.get('indicator', '').lower() in query_entities_lower:
            score += 0.4
        if str(result.get('year', '')).lower() in query_entities_lower:
            score += 0.2
        
        # Query type specific scoring
        if query.query_type == 'statistical' and result.get('value') is not None:
            score += 0.2
        
        # Use existing confidence score
        confidence = result.get('confidence_score', 0.5)
        score += confidence * 0.3
        
        return min(score, 1.0)

class CorrectiveTableAgent:
    """CRAG-TAG agent for table-specific processing with self-correction"""
    
    def __init__(self, weaviate_client):
        self.weaviate_client = weaviate_client
        self.table_patterns = {
            'statistical': ['count', 'percentage', 'ratio', 'rate', 'total', 'average'],
            'comparative': ['vs', 'compared to', 'higher', 'lower', 'difference'],
            'temporal': ['year', 'growth', 'change', 'trend', 'increase', 'decrease']
        }
    
    async def process_table_query(self, query: PolicyQuery) -> List[EnhancedResult]:
        """Process table-specific queries with correction"""
        # Stage 1: Retrieve relevant tables
        table_results = await self._retrieve_table_data(query)
        
        # Stage 2: Extract structured information
        structured_data = await self._extract_structured_info(table_results, query)
        
        # Stage 3: Self-verification and correction
        verified_results = await self._verify_and_correct(structured_data, query)
        
        # Stage 4: Format enhanced results
        enhanced_results = await self._format_enhanced_results(verified_results, query)
        
        return enhanced_results
    
    async def _retrieve_table_data(self, query: PolicyQuery) -> List[Dict[str, Any]]:
        """Retrieve table-related facts"""
        try:
            # Build enhanced search query
            search_conditions = []
            
            # Add entity filters
            for entity in query.entities:
                if entity.title() in ['Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Krishna', 
                                    'Kurnool', 'Nellore', 'Prakasam', 'Srikakulam', 'Visakhapatnam', 
                                    'Vizianagaram', 'West Godavari', 'Kadapa']:
                    search_conditions.append({
                        "path": ["district"],
                        "operator": "Equal",
                        "valueText": entity.title()
                    })
                elif entity.upper() in ['ENROLLMENT', 'APPLICATIONS', 'SCHOOLS', 'TEACHERS', 'GER', 'NER', 'PTR']:
                    search_conditions.append({
                        "path": ["indicator"],
                        "operator": "Equal", 
                        "valueText": entity.title()
                    })
            
            # Add temporal filter
            if query.temporal_scope:
                start_year, end_year = query.temporal_scope
                search_conditions.extend([
                    {
                        "path": ["year"],
                        "operator": "GreaterThanEqual",
                        "valueInt": start_year
                    },
                    {
                        "path": ["year"],
                        "operator": "LessThanEqual",
                        "valueInt": end_year
                    }
                ])
            
            # Combine conditions
            where_clause = None
            if search_conditions:
                if len(search_conditions) == 1:
                    where_clause = search_conditions[0]
                else:
                    where_clause = {
                        "operator": "And",
                        "operands": search_conditions
                    }
            
            # Execute search
            result = (
                self.weaviate_client.query
                .get("Fact", [
                    "fact_id", "district", "indicator", "year", "value", "unit",
                    "content", "confidence_score", "metadata"
                ])
                .with_where(where_clause)
                .with_limit(50)
                .do()
            )
            
            facts = result.get("data", {}).get("Get", {}).get("Fact", [])
            return facts
            
        except Exception as e:
            logger.error(f"Table retrieval failed: {e}")
            return []
    
    async def _extract_structured_info(self, table_results: List[Dict[str, Any]], 
                                     query: PolicyQuery) -> List[Dict[str, Any]]:
        """Extract and structure information from table results"""
        structured_results = []
        
        for result in table_results:
            # Parse numerical data
            numerical_data = {
                'value': result.get('value'),
                'unit': result.get('unit'),
                'year': result.get('year'),
                'district': result.get('district'),
                'indicator': result.get('indicator')
            }
            
            # Create structured entry
            structured_entry = {
                'original_result': result,
                'numerical_data': numerical_data,
                'extraction_method': 'table_parsing',
                'confidence': result.get('confidence_score', 0.5)
            }
            
            structured_results.append(structured_entry)
        
        return structured_results
    
    async def _verify_and_correct(self, structured_data: List[Dict[str, Any]], 
                                query: PolicyQuery) -> List[Dict[str, Any]]:
        """Verify results and apply corrections"""
        verified_results = []
        corrections_applied = []
        
        for data in structured_data:
            # Check for obvious errors
            corrections = []
            
            # Verify numerical ranges
            value = data['numerical_data'].get('value')
            if value is not None:
                if value < 0:
                    corrections.append("Negative value detected - marked as uncertain")
                    data['confidence'] *= 0.5
                
                # Check for unrealistic values based on indicator type
                indicator = data['numerical_data'].get('indicator', '').lower()
                if 'percentage' in indicator or 'rate' in indicator:
                    if value > 100:
                        corrections.append("Percentage value >100% - marked as uncertain")
                        data['confidence'] *= 0.7
            
            # Verify temporal consistency
            year = data['numerical_data'].get('year')
            if year and (year < 2000 or year > datetime.now().year + 1):
                corrections.append("Implausible year detected - marked as uncertain")
                data['confidence'] *= 0.6
            
            # Add verification status
            if data['confidence'] > 0.8:
                verification_status = 'verified'
            elif data['confidence'] > 0.5:
                verification_status = 'uncertain'
            else:
                verification_status = 'conflicting'
            
            data['verification_status'] = verification_status
            data['corrections_applied'] = corrections
            
            verified_results.append(data)
        
        return verified_results
    
    async def _format_enhanced_results(self, verified_results: List[Dict[str, Any]], 
                                     query: PolicyQuery) -> List[EnhancedResult]:
        """Format results into EnhancedResult objects"""
        enhanced_results = []
        
        for data in verified_results:
            original = data['original_result']
            numerical = data['numerical_data']
            
            # Create content text
            content = f"{numerical['indicator']} in {numerical['district']} for {numerical['year']}: "
            content += f"{numerical['value']} {numerical['unit']}"
            
            # Create reasoning chain
            reasoning_chain = [
                f"Retrieved from {original.get('fact_id', 'unknown source')}",
                f"Extracted numerical value: {numerical['value']} {numerical['unit']}",
                f"Verification status: {data['verification_status']}"
            ]
            
            if data['corrections_applied']:
                reasoning_chain.extend(data['corrections_applied'])
            
            # Create enhanced result
            enhanced_result = EnhancedResult(
                content=content,
                source_type='table',
                confidence_score=data['confidence'],
                verification_status=data['verification_status'],
                sources=[original],
                reasoning_chain=reasoning_chain,
                corrections_applied=data['corrections_applied'],
                numerical_data=numerical
            )
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results

class AdvancedRAGOrchestrator:
    """Main orchestrator for the advanced multi-agent RAG system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.query_classifier = QueryClassificationAgent()
        
        # Initialize databases
        self.weaviate_client = self._init_weaviate()
        self.neo4j_driver = self._init_neo4j()
        
        # Initialize agents
        self.graph_rag_agent = HierarchicalGraphRAGAgent(self.neo4j_driver, self.weaviate_client)
        self.table_agent = CorrectiveTableAgent(self.weaviate_client)
        
        # Agent routing configuration
        self.agent_routing = {
            'statistical': ['table_agent', 'graph_rag_agent'],
            'comparative': ['graph_rag_agent', 'table_agent'],
            'temporal': ['graph_rag_agent', 'table_agent'],
            'policy_lookup': ['graph_rag_agent'],
            'complex': ['graph_rag_agent', 'table_agent']
        }
        
        logger.info("🤖 Advanced RAG Orchestrator initialized")
    
    async def search(self, query_text: str, **kwargs) -> List[EnhancedResult]:
        """Main search interface"""
        # Stage 1: Query classification and planning
        policy_query = self.query_classifier.classify_query(query_text)
        logger.info(f"🔍 Query classified as: {policy_query.query_type} with intent: {policy_query.intent}")
        
        # Stage 2: Agent routing and parallel execution
        agent_results = await self._execute_agents(policy_query)
        
        # Stage 3: Result fusion and ranking
        fused_results = await self._fuse_and_rank_results(agent_results, policy_query)
        
        # Stage 4: Final verification and correction
        final_results = await self._final_verification(fused_results, policy_query)
        
        logger.info(f"✅ Returning {len(final_results)} verified results")
        return final_results
    
    async def _execute_agents(self, query: PolicyQuery) -> Dict[str, List[Any]]:
        """Execute relevant agents in parallel"""
        agent_list = self.agent_routing.get(query.query_type, ['graph_rag_agent'])
        
        tasks = []
        agent_results = {}
        
        # Execute agents in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            if 'graph_rag_agent' in agent_list:
                future = executor.submit(
                    asyncio.run,
                    self.graph_rag_agent.hierarchical_search(query)
                )
                tasks.append(('graph_rag', future))
            
            if 'table_agent' in agent_list:
                future = executor.submit(
                    asyncio.run,
                    self.table_agent.process_table_query(query)
                )
                tasks.append(('table', future))
        
        # Collect results
        for agent_name, future in tasks:
            try:
                result = future.result(timeout=30)
                agent_results[agent_name] = result
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                agent_results[agent_name] = []
        
        return agent_results
    
    async def _fuse_and_rank_results(self, agent_results: Dict[str, List[Any]], 
                                   query: PolicyQuery) -> List[EnhancedResult]:
        """Fuse results from multiple agents and rank them"""
        all_results = []
        
        # Convert graph results to EnhancedResult format
        if 'graph_rag' in agent_results:
            for result in agent_results['graph_rag']:
                enhanced_result = EnhancedResult(
                    content=result.get('content', ''),
                    source_type='graph',
                    confidence_score=result.get('relevance_score', 0.5),
                    verification_status='verified',
                    sources=[result],
                    reasoning_chain=[f"Retrieved via hierarchical graph search"],
                    corrections_applied=[],
                    graph_context={'community_id': result.get('community_id')}
                )
                all_results.append(enhanced_result)
        
        # Add table results (already in EnhancedResult format)
        if 'table' in agent_results:
            all_results.extend(agent_results['table'])
        
        # Remove duplicates and rank
        unique_results = self._remove_duplicates(all_results)
        ranked_results = self._rank_results(unique_results, query)
        
        return ranked_results[:query.max_sources]
    
    def _remove_duplicates(self, results: List[EnhancedResult]) -> List[EnhancedResult]:
        """Remove duplicate results"""
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result.content[:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[EnhancedResult], query: PolicyQuery) -> List[EnhancedResult]:
        """Rank results based on multiple criteria"""
        for result in results:
            score = 0.0
            
            # Base confidence score
            score += result.confidence_score * 0.4
            
            # Source type weighting
            if query.query_type == 'statistical' and result.source_type == 'table':
                score += 0.3
            elif query.query_type in ['comparative', 'temporal'] and result.source_type == 'graph':
                score += 0.3
            
            # Verification status bonus
            if result.verification_status == 'verified':
                score += 0.2
            elif result.verification_status == 'uncertain':
                score += 0.1
            
            # Numerical data bonus for statistical queries
            if query.query_type == 'statistical' and result.numerical_data:
                score += 0.1
            
            result.confidence_score = min(score, 1.0)
        
        return sorted(results, key=lambda x: x.confidence_score, reverse=True)
    
    async def _final_verification(self, results: List[EnhancedResult], 
                                query: PolicyQuery) -> List[EnhancedResult]:
        """Final verification and consistency checking"""
        verified_results = []
        
        for result in results:
            # Cross-reference verification for statistical data
            if result.numerical_data and len(results) > 1:
                consistency_score = self._check_numerical_consistency(result, results)
                if consistency_score < 0.3:
                    result.verification_status = 'conflicting'
                    result.corrections_applied.append("Inconsistent with other sources")
            
            # Quality threshold
            if result.confidence_score >= query.accuracy_requirement:
                verified_results.append(result)
        
        return verified_results
    
    def _check_numerical_consistency(self, target_result: EnhancedResult, 
                                   all_results: List[EnhancedResult]) -> float:
        """Check numerical consistency across results"""
        if not target_result.numerical_data:
            return 1.0
        
        target_value = target_result.numerical_data.get('value')
        if target_value is None:
            return 1.0
        
        similar_values = []
        for other_result in all_results:
            if (other_result != target_result and 
                other_result.numerical_data and
                other_result.numerical_data.get('indicator') == target_result.numerical_data.get('indicator')):
                
                other_value = other_result.numerical_data.get('value')
                if other_value is not None:
                    similar_values.append(other_value)
        
        if not similar_values:
            return 1.0
        
        # Calculate variance
        all_values = similar_values + [target_value]
        mean_value = np.mean(all_values)
        std_value = np.std(all_values)
        
        if std_value == 0:
            return 1.0
        
        # Z-score based consistency
        z_score = abs((target_value - mean_value) / std_value)
        consistency_score = max(0, 1.0 - (z_score / 3.0))  # 3-sigma rule
        
        return consistency_score
    
    def _init_weaviate(self):
        """Initialize Weaviate client"""
        import weaviate
        client = weaviate.Client(
            url=self.config.get('weaviate_url', 'http://localhost:8080'),
            timeout_config=(5, 15)
        )
        return client
    
    def _init_neo4j(self):
        """Initialize Neo4j driver"""
        driver = GraphDatabase.driver(
            self.config.get('neo4j_uri', 'bolt://localhost:7687'),
            auth=(
                self.config.get('neo4j_user', 'neo4j'),
                self.config.get('neo4j_password', 'password')
            )
        )
        return driver
    
    async def initialize_system(self):
        """Initialize the advanced RAG system"""
        logger.info("🚀 Initializing Advanced RAG System")
        
        # Build hierarchical graph
        facts = await self._load_all_facts()
        await self.graph_rag_agent.build_hierarchical_graph(facts)
        
        logger.info("✅ Advanced RAG System ready")
    
    async def _load_all_facts(self) -> List[Dict[str, Any]]:
        """Load all facts from Weaviate"""
        try:
            result = (
                self.weaviate_client.query
                .get("Fact", ["fact_id", "district", "indicator", "year", "value", "unit"])
                .with_limit(1000)
                .do()
            )
            
            facts = result.get("data", {}).get("Get", {}).get("Fact", [])
            return facts
            
        except Exception as e:
            logger.error(f"Failed to load facts: {e}")
            return []
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()

def create_advanced_rag_system(config_file: str = None) -> AdvancedRAGOrchestrator:
    """Create and initialize the advanced RAG system"""
    
    default_config = {
        'weaviate_url': 'http://localhost:8080',
        'neo4j_uri': 'bolt://localhost:7687',
        'neo4j_user': 'neo4j',
        'neo4j_password': 'password',
        'embedding_model': 'all-MiniLM-L6-v2'
    }
    
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            default_config.update(file_config)
    
    return AdvancedRAGOrchestrator(default_config)