"""
Gemini-powered RAG Service with Citation Support
Integrates Google Gemini API for enhanced response generation with source citations
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import re
import google.generativeai as genai
from backend.retriever import WeaviateRetriever

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Citation metadata for source tracking"""
    id: str
    source_document: str
    district: str
    indicator: str
    year: int
    page_number: Optional[int] = None
    confidence_score: float = 0.0
    excerpt: str = ""
    url: Optional[str] = None

@dataclass
class RAGResponse:
    """RAG response with citations"""
    answer: str
    citations: List[Citation]
    query: str
    retrieval_stats: Dict[str, Any]
    confidence_score: float

class GeminiRAGService:
    """RAG service using Google Gemini API with citation tracking"""
    
    def __init__(self, api_key: str):
        """Initialize Gemini RAG service"""
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize retriever
        self.retriever = WeaviateRetriever()
        
        # Citation tracking
        self.citation_cache = {}
        
        logger.info("Initialized Gemini RAG service")
    
    def _create_citation_id(self, source_info: Dict[str, Any]) -> str:
        """Create unique citation ID"""
        content = f"{source_info.get('source_document', '')}{source_info.get('district', '')}{source_info.get('indicator', '')}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _extract_citations_from_results(self, search_results: List[Dict[str, Any]]) -> List[Citation]:
        """Convert search results to citations"""
        citations = []
        
        for i, result in enumerate(search_results):
            # Extract relevant text excerpt
            excerpt = ""
            if 'content' in result:
                excerpt = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            elif 'text' in result:
                excerpt = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            
            citation = Citation(
                id=self._create_citation_id(result),
                source_document=result.get('source_document', 'Unknown'),
                district=result.get('district', 'N/A'),
                indicator=result.get('indicator', 'N/A'),
                year=result.get('year', 0),
                page_number=result.get('page_number'),
                confidence_score=result.get('_additional', {}).get('score', 0.0),
                excerpt=excerpt
            )
            
            citations.append(citation)
            
        return citations
    
    def _format_context_for_gemini(self, search_results: List[Dict[str, Any]], query: str) -> str:
        """Format retrieved context for Gemini prompt"""
        context_parts = []
        
        context_parts.append("Based on the following documents from the Andhra Pradesh education policy database:")
        context_parts.append("")
        
        for i, result in enumerate(search_results, 1):
            # Format each document
            doc_info = f"Document {i}:"
            doc_info += f"\nSource: {result.get('source_document', 'Unknown')}"
            doc_info += f"\nDistrict: {result.get('district', 'N/A')}"
            doc_info += f"\nIndicator: {result.get('indicator', 'N/A')}"
            doc_info += f"\nYear: {result.get('year', 'N/A')}"
            
            # Add content
            content = result.get('content', result.get('text', ''))
            if content:
                doc_info += f"\nContent: {content}"
            
            context_parts.append(doc_info)
            context_parts.append("-" * 50)
        
        return "\n".join(context_parts)
    
    def _create_gemini_prompt(self, query: str, context: str, citations: List[Citation]) -> str:
        """Create comprehensive prompt for Gemini"""
        
        citation_refs = "\n".join([
            f"[{i+1}] {cite.source_document} - {cite.district} ({cite.year}) - {cite.indicator}"
            for i, cite in enumerate(citations)
        ])
        
        prompt = f"""You are an AI assistant specialized in Andhra Pradesh education policy analysis. 

CONTEXT:
{context}

QUERY: {query}

CITATION REFERENCES:
{citation_refs}

INSTRUCTIONS:
1. Provide a comprehensive answer based ONLY on the provided context
2. Use specific data, statistics, and facts from the documents
3. Include citation numbers [1], [2], etc. throughout your response to reference specific sources
4. If information is not available in the context, clearly state this
5. Focus on actionable insights relevant to AP education policy
6. Structure your response clearly with headings if appropriate
7. Highlight key statistics, trends, and policy implications

RESPONSE FORMAT:
- Start with a direct answer to the query
- Provide supporting details with citations
- Include relevant statistics and data points
- End with key takeaways or recommendations if applicable

Remember: Only use information from the provided documents and always include citation numbers."""

        return prompt
    
    def query(
        self,
        user_query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_citations: bool = True
    ) -> RAGResponse:
        """Process RAG query with Gemini and return response with citations"""
        
        logger.info(f"Processing query: {user_query}")
        
        try:
            # Step 1: Retrieve relevant documents
            search_results = self.retriever.search(
                query=user_query,
                limit=limit,
                filters=filters
            )
            
            logger.info(f"Retrieved {len(search_results)} documents")
            
            if not search_results:
                return RAGResponse(
                    answer="I couldn't find any relevant information in the database for your query. Please try rephrasing your question or check if the topic is covered in our policy documents.",
                    citations=[],
                    query=user_query,
                    retrieval_stats={"total_results": 0},
                    confidence_score=0.0
                )
            
            # Step 2: Create citations from search results
            citations = self._extract_citations_from_results(search_results) if include_citations else []
            
            # Step 3: Format context for Gemini
            context = self._format_context_for_gemini(search_results, user_query)
            
            # Step 4: Create Gemini prompt
            prompt = self._create_gemini_prompt(user_query, context, citations)
            
            # Step 5: Generate response with Gemini
            logger.info("Generating response with Gemini...")
            response = self.model.generate_content(prompt)
            
            if not response.text:
                raise Exception("Gemini returned empty response")
            
            # Step 6: Calculate overall confidence
            avg_confidence = sum(cite.confidence_score for cite in citations) / len(citations) if citations else 0.0
            
            # Step 7: Create RAG response
            rag_response = RAGResponse(
                answer=response.text,
                citations=citations,
                query=user_query,
                retrieval_stats={
                    "total_results": len(search_results),
                    "avg_confidence": avg_confidence,
                    "filters_applied": bool(filters)
                },
                confidence_score=avg_confidence
            )
            
            logger.info(f"Generated response with {len(citations)} citations")
            return rag_response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return RAGResponse(
                answer=f"I encountered an error while processing your query: {str(e)}. Please try again or contact support.",
                citations=[],
                query=user_query,
                retrieval_stats={"error": str(e)},
                confidence_score=0.0
            )
    
    def format_citations_for_display(self, citations: List[Citation]) -> List[Dict[str, Any]]:
        """Format citations for UI display"""
        formatted_citations = []
        
        for i, citation in enumerate(citations, 1):
            formatted_citation = {
                "number": i,
                "id": citation.id,
                "title": f"{citation.source_document} - {citation.indicator}",
                "source": citation.source_document,
                "district": citation.district,
                "year": citation.year,
                "indicator": citation.indicator,
                "excerpt": citation.excerpt,
                "confidence": f"{citation.confidence_score:.2f}",
                "page": citation.page_number if citation.page_number else "N/A"
            }
            formatted_citations.append(formatted_citation)
        
        return formatted_citations
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            # Test Gemini API
            test_response = self.model.generate_content("Hello")
            gemini_status = "online" if test_response.text else "error"
            
            # Test retriever
            test_search = self.retriever.search("test", limit=1)
            retriever_status = "online" if isinstance(test_search, list) else "error"
            
            return {
                "status": "healthy" if gemini_status == "online" and retriever_status == "online" else "degraded",
                "gemini_api": gemini_status,
                "weaviate_retriever": retriever_status,
                "timestamp": "now"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": "now"
            }

# Example usage
if __name__ == "__main__":
    # Test the service
    api_key = "YOUR_GEMINI_API_KEY"  # Replace with actual key
    service = GeminiRAGService(api_key)
    
    # Test query
    response = service.query("What are the enrollment statistics for Krishna district?")
    
    print("Answer:", response.answer)
    print("Citations:", len(response.citations))
    for citation in response.citations:
        print(f"- {citation.source_document} ({citation.year}) - {citation.district}")