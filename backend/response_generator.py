#!/usr/bin/env python3
"""
Citation-Aware Response Generator for AP Policy Co-Pilot
Constrains LLM to only paraphrase information from retrieved documents
"""
import os
import logging
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """LLM generation configuration"""
    provider: str  # 'gemini', 'claude', 'gpt4'
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str

class CitationAwareGenerator:
    """
    Response generator that enforces citation-first architecture.
    LLM can only paraphrase information from provided context.
    """
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
        enforce_citations: bool = True
    ):
        self.config = GenerationConfig(
            provider=llm_config.get('provider', 'gemini'),
            model=llm_config.get('model', 'gemini-2.0-flash-exp'),
            temperature=llm_config.get('temperature', 0.0),
            max_tokens=llm_config.get('max_tokens', 2000),
            system_prompt=llm_config.get('system_prompt', self._default_system_prompt())
        )
        self.enforce_citations = enforce_citations
        
        # Initialize LLM client
        self.client = self._init_llm_client()
        
        logger.info(f"✅ Response Generator initialized (provider: {self.config.provider})")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for citation-first generation"""
        return """You are a policy reasoning assistant for the Andhra Pradesh School Education Department.

CRITICAL RULES:
1. ONLY use information from the provided context documents
2. NEVER invent, infer, or elaborate beyond what is explicitly stated in the sources
3. Every factual statement must be traceable to a specific citation
4. If information is not in the context, say "This information is not available in the provided documents"
5. Use precise language and quote specific sections/clauses when referencing legal documents
6. For statistics, include the year and source
7. Maintain a professional, objective tone
8. Do not add personal opinions or interpretations

Your role is to accurately paraphrase and synthesize information from official government documents, not to generate new content."""
    
    def _init_llm_client(self):
        """Initialize LLM client based on provider"""
        if self.config.provider == 'gemini':
            return self._init_gemini()
        elif self.config.provider == 'claude':
            return self._init_claude()
        elif self.config.provider == 'gpt4':
            return self._init_gpt4()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    def _init_gemini(self):
        """Initialize Google Gemini client"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            genai.configure(api_key=api_key)
            
            # Configure model
            generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            }
            
            model = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config=generation_config,
                system_instruction=self.config.system_prompt
            )
            
            logger.info(f"   ✅ Gemini client initialized: {self.config.model}")
            return model
            
        except Exception as e:
            logger.error(f"   ❌ Gemini initialization failed: {e}")
            raise
    
    def _init_claude(self):
        """Initialize Anthropic Claude client"""
        try:
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
            client = anthropic.Anthropic(api_key=api_key)
            
            logger.info(f"   ✅ Claude client initialized: {self.config.model}")
            return client
            
        except Exception as e:
            logger.error(f"   ❌ Claude initialization failed: {e}")
            raise
    
    def _init_gpt4(self):
        """Initialize OpenAI GPT-4 client"""
        try:
            import openai
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            openai.api_key = api_key
            
            logger.info(f"   ✅ GPT-4 client initialized: {self.config.model}")
            return openai
            
        except Exception as e:
            logger.error(f"   ❌ GPT-4 initialization failed: {e}")
            raise
    
    async def generate(
        self,
        query: str,
        context: str,
        legal_chain: Optional[List[str]] = None,
        citations: Optional[List[Any]] = None
    ) -> str:
        """
        Generate response using LLM with citation constraints.
        
        Args:
            query: User's policy question
            context: Retrieved document context
            legal_chain: Legal hierarchy (Act → Rule → GO)
            citations: List of citation objects
        
        Returns:
            Generated response text
        """
        logger.info(f"🤖 Generating response for: '{query[:50]}...'")
        
        # Build prompt with strict constraints
        prompt = self._build_prompt(query, context, legal_chain, citations)
        
        # Generate response
        if self.config.provider == 'gemini':
            response_text = await self._generate_gemini(prompt)
        elif self.config.provider == 'claude':
            response_text = await self._generate_claude(prompt)
        elif self.config.provider == 'gpt4':
            response_text = await self._generate_gpt4(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        # Post-process to enforce citations
        if self.enforce_citations:
            response_text = self._enforce_citation_format(response_text)
        
        logger.info(f"   ✅ Generated {len(response_text)} characters")
        return response_text
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        legal_chain: Optional[List[str]],
        citations: Optional[List[Any]]
    ) -> str:
        """Build prompt with context and constraints"""
        prompt_parts = []
        
        # Query
        prompt_parts.append(f"USER QUERY:\n{query}\n")
        
        # Legal chain
        if legal_chain:
            prompt_parts.append(f"LEGAL HIERARCHY:")
            for item in legal_chain:
                prompt_parts.append(f"  → {item}")
            prompt_parts.append("")
        
        # Context documents
        prompt_parts.append("AVAILABLE CONTEXT:")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Citations list
        if citations:
            prompt_parts.append("CITATION SOURCES:")
            for i, citation in enumerate(citations, 1):
                if hasattr(citation, 'to_citation_string'):
                    prompt_parts.append(f"[{i}] {citation.to_citation_string()}")
                elif hasattr(citation, 'title'):
                    prompt_parts.append(f"[{i}] {getattr(citation, 'title', 'Unknown')}")
            prompt_parts.append("")
        
        # Instructions
        prompt_parts.append("INSTRUCTIONS:")
        prompt_parts.append("1. Answer the query using ONLY information from the context above")
        prompt_parts.append("2. Be precise and factual - do not elaborate or infer")
        prompt_parts.append("3. Reference specific sections/clauses when citing legal documents")
        prompt_parts.append("4. Include years and sources for all statistics")
        prompt_parts.append("5. If information is missing, explicitly state that")
        prompt_parts.append("6. Keep the response concise and directly relevant")
        prompt_parts.append("")
        prompt_parts.append("YOUR RESPONSE:")
        
        return "\n".join(prompt_parts)
    
    async def _generate_gemini(self, prompt: str) -> str:
        """Generate using Gemini"""
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    async def _generate_claude(self, prompt: str) -> str:
        """Generate using Claude"""
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.config.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            raise
    
    async def _generate_gpt4(self, prompt: str) -> str:
        """Generate using GPT-4"""
        try:
            response = await self.client.ChatCompletion.acreate(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT-4 generation failed: {e}")
            raise
    
    def _enforce_citation_format(self, text: str) -> str:
        """Post-process to ensure proper citation format"""
        # This is a placeholder - actual implementation would:
        # 1. Parse the response
        # 2. Identify factual claims
        # 3. Add inline citations if missing
        # 4. Format citations consistently
        
        # For now, just ensure clean text
        text = text.strip()
        
        # Remove any markdown/formatting artifacts
        text = text.replace('**', '')
        text = text.replace('__', '')
        
        return text
    
    def generate_fallback_response(
        self,
        query: str,
        reason: str = "Unable to find relevant information"
    ) -> str:
        """Generate fallback response when retrieval fails"""
        return (
            f"I apologize, but I cannot provide a comprehensive answer to your query: '{query}'\n\n"
            f"Reason: {reason}\n\n"
            f"Please try:\n"
            f"1. Rephrasing your question with more specific terms\n"
            f"2. Checking if the information is available in the document corpus\n"
            f"3. Contacting the AP School Education Department directly for this information"
        )

if __name__ == "__main__":
    # Test generator
    import asyncio
    
    async def test():
        config = {
            'provider': 'gemini',
            'model': 'gemini-2.0-flash-exp',
            'temperature': 0.0,
            'max_tokens': 1000
        }
        
        generator = CitationAwareGenerator(config, enforce_citations=True)
        
        # Mock context
        context = """
        ### Legal Framework:
        - AP Education Act 1982: Section 12 establishes School Management Committees
        - AP RTE Rules 2010: Rule 4(3) defines SMC composition and responsibilities
        
        ### Statistical Data:
        - Dropout rate SC students: 15.2% (Anantapur, 2016-17)
        - Source: UDISE+ 2016-17 State Report
        """
        
        legal_chain = [
            "Act: AP Education Act 1982",
            "Rule: AP RTE Rules 2010"
        ]
        
        # Generate
        response = await generator.generate(
            query="What are the responsibilities of School Management Committees?",
            context=context,
            legal_chain=legal_chain
        )
        
        print("\n=== Generated Response ===")
        print(response)
    
    asyncio.run(test())
