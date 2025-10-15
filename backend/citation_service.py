#!/usr/bin/env python3
"""
Citation Validator for AP Policy Co-Pilot
Ensures non-hallucination by validating every claim against retrieved documents
"""
import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Citation validation result"""
    valid: bool
    confidence: float
    errors: List[str]
    warnings: List[str]
    claim_citation_map: Dict[str, List[str]]  # Map each claim to its citations
    uncited_claims: List[str]
    hallucination_risk: float

class CitationValidator:
    """
    Validates that responses only contain information from retrieved documents.
    Prevents hallucination by checking semantic similarity between claims and sources.
    """
    
    def __init__(
        self,
        min_citations: int = 1,
        validate_chain: bool = True,
        similarity_threshold: float = 0.6,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.min_citations = min_citations
        self.validate_chain = validate_chain
        self.similarity_threshold = similarity_threshold
        
        # Initialize embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(embedding_model)
        
        logger.info(f"✅ Citation Validator initialized (threshold: {similarity_threshold})")
    
    async def validate(
        self,
        response_text: str,
        citations: List[Any],  # List of Citation objects
        legal_chain: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main validation method.
        
        Checks:
        1. Every claim has at least one citation
        2. Citations are semantically similar to claims
        3. Legal chain is valid (Act → Rule → GO)
        4. No hallucinated information
        """
        logger.info("🔍 Validating citations...")
        
        errors = []
        warnings = []
        
        # Step 1: Extract claims from response
        claims = self._extract_claims(response_text)
        logger.info(f"   Extracted {len(claims)} claims from response")
        
        # Step 2: Extract citation texts
        citation_texts = self._extract_citation_texts(citations)
        logger.info(f"   Found {len(citation_texts)} citation sources")
        
        # Step 3: Map claims to citations using semantic similarity
        claim_citation_map, uncited_claims = self._map_claims_to_citations(
            claims, citation_texts
        )
        
        # Step 4: Check citation coverage
        if uncited_claims:
            errors.append(
                f"{len(uncited_claims)} claims lack supporting citations: "
                f"{uncited_claims[:3]}"  # Show first 3
            )
        
        if len(citations) < self.min_citations:
            errors.append(
                f"Insufficient citations: {len(citations)} provided, "
                f"{self.min_citations} required"
            )
        
        # Step 5: Validate legal chain
        if self.validate_chain and legal_chain:
            chain_valid, chain_errors = self._validate_legal_chain(legal_chain, citations)
            if not chain_valid:
                errors.extend(chain_errors)
        
        # Step 6: Detect potential hallucinations
        hallucination_risk, hallucination_warnings = self._detect_hallucinations(
            response_text, citation_texts
        )
        warnings.extend(hallucination_warnings)
        
        # Step 7: Check for weasel words/hedging
        hedging_warnings = self._check_hedging(response_text)
        warnings.extend(hedging_warnings)
        
        # Calculate overall confidence
        citation_coverage = (len(claims) - len(uncited_claims)) / max(len(claims), 1)
        confidence = (
            citation_coverage * 0.5 +
            (1 - hallucination_risk) * 0.3 +
            min(len(citations) / self.min_citations, 1.0) * 0.2
        )
        
        # Determine validity
        valid = len(errors) == 0 and hallucination_risk < 0.3
        
        logger.info(f"   Validation: {'✅ PASS' if valid else '❌ FAIL'} "
                   f"(confidence: {confidence:.2%}, risk: {hallucination_risk:.2%})")
        
        return {
            'valid': valid,
            'confidence': confidence,
            'errors': errors,
            'warnings': warnings,
            'claim_citation_map': claim_citation_map,
            'uncited_claims': uncited_claims,
            'hallucination_risk': hallucination_risk,
            'citation_coverage': citation_coverage,
            'total_claims': len(claims),
            'total_citations': len(citations)
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from response text"""
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip empty, very short, or non-factual sentences
            if len(sentence) < 20:
                continue
            
            # Skip meta-sentences (e.g., "According to...", "As stated...")
            if any(phrase in sentence.lower() for phrase in [
                'according to', 'as stated', 'as mentioned', 'as per',
                'the document states', 'the following', 'in summary'
            ]):
                continue
            
            # Skip questions
            if sentence.strip().endswith('?'):
                continue
            
            claims.append(sentence)
        
        return claims
    
    def _extract_citation_texts(self, citations: List[Any]) -> List[str]:
        """Extract text content from citation objects"""
        citation_texts = []
        
        for citation in citations:
            # Handle different citation formats
            if hasattr(citation, 'excerpt'):
                citation_texts.append(citation.excerpt)
            elif hasattr(citation, 'text'):
                citation_texts.append(citation.text)
            elif isinstance(citation, dict):
                if 'excerpt' in citation:
                    citation_texts.append(citation['excerpt'])
                elif 'text' in citation:
                    citation_texts.append(citation['text'])
                elif 'full_text' in citation:
                    # Truncate long texts
                    citation_texts.append(citation['full_text'][:1000])
        
        return citation_texts
    
    def _map_claims_to_citations(
        self,
        claims: List[str],
        citation_texts: List[str]
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """Map claims to supporting citations using semantic similarity"""
        
        if not claims or not citation_texts:
            return {}, claims
        
        # Generate embeddings
        claim_embeddings = self.embedding_model.encode(claims)
        citation_embeddings = self.embedding_model.encode(citation_texts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(claim_embeddings, citation_embeddings)
        
        claim_citation_map = {}
        uncited_claims = []
        
        for i, claim in enumerate(claims):
            # Find citations above similarity threshold
            similarities = similarity_matrix[i]
            supporting_citations = [
                citation_texts[j]
                for j, sim in enumerate(similarities)
                if sim >= self.similarity_threshold
            ]
            
            if supporting_citations:
                claim_citation_map[claim] = supporting_citations
            else:
                uncited_claims.append(claim)
                # Log low similarity
                max_sim = max(similarities)
                logger.warning(
                    f"Uncited claim: '{claim[:50]}...' "
                    f"(max similarity: {max_sim:.2f})"
                )
        
        return claim_citation_map, uncited_claims
    
    def _validate_legal_chain(
        self,
        legal_chain: List[str],
        citations: List[Any]
    ) -> Tuple[bool, List[str]]:
        """Validate that legal chain follows proper hierarchy (Act → Rule → GO)"""
        errors = []
        
        # Expected order
        expected_order = ['act', 'rule', 'go']
        
        # Extract document types from chain
        chain_types = []
        for item in legal_chain:
            item_lower = item.lower()
            if 'act' in item_lower:
                chain_types.append('act')
            elif 'rule' in item_lower:
                chain_types.append('rule')
            elif 'go' in item_lower or 'government order' in item_lower:
                chain_types.append('go')
        
        # Check order
        for i in range(len(chain_types) - 1):
            current_idx = expected_order.index(chain_types[i])
            next_idx = expected_order.index(chain_types[i + 1])
            
            if next_idx < current_idx:
                errors.append(
                    f"Invalid legal chain order: {chain_types[i]} cannot come before {chain_types[i+1]}"
                )
        
        # Verify citations match chain
        citation_types = set()
        for citation in citations:
            if hasattr(citation, 'doc_type'):
                doc_type = str(citation.doc_type).lower()
                if 'act' in doc_type:
                    citation_types.add('act')
                elif 'rule' in doc_type:
                    citation_types.add('rule')
                elif 'go' in doc_type:
                    citation_types.add('go')
        
        chain_type_set = set(chain_types)
        if not citation_types.issuperset(chain_type_set):
            missing = chain_type_set - citation_types
            errors.append(
                f"Legal chain references documents not present in citations: {missing}"
            )
        
        return len(errors) == 0, errors
    
    def _detect_hallucinations(
        self,
        response_text: str,
        citation_texts: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Detect potential hallucinations by checking for specific content
        that commonly indicates unsupported claims.
        """
        warnings = []
        risk_score = 0.0
        
        # Combine all citation text
        citation_corpus = ' '.join(citation_texts).lower()
        response_lower = response_text.lower()
        
        # Check for specific numbers/dates not in citations
        response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', response_text))
        citation_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', ' '.join(citation_texts)))
        
        unsupported_numbers = response_numbers - citation_numbers
        if unsupported_numbers and len(unsupported_numbers) > 2:
            warnings.append(
                f"Response contains {len(unsupported_numbers)} numbers not found in citations: "
                f"{list(unsupported_numbers)[:5]}"
            )
            risk_score += 0.2
        
        # Check for definitive statements without hedging
        definitive_patterns = [
            r'\b(always|never|all|none|every|must|will definitely)\b',
            r'\b(it is certain|without doubt|undoubtedly)\b'
        ]
        for pattern in definitive_patterns:
            if re.search(pattern, response_lower):
                warnings.append(
                    f"Definitive statement without hedging may indicate unsupported claim"
                )
                risk_score += 0.1
                break
        
        # Check for common hallucination phrases
        hallucination_phrases = [
            'it is well known that',
            'studies show that',
            'research indicates',
            'experts agree',
            'it is generally accepted',
            'common knowledge'
        ]
        for phrase in hallucination_phrases:
            if phrase in response_lower:
                if phrase not in citation_corpus:
                    warnings.append(
                        f"Generic claim '{phrase}' not supported by citations"
                    )
                    risk_score += 0.15
        
        # Check response length vs citation length
        response_words = len(response_text.split())
        citation_words = len(citation_corpus.split())
        
        if response_words > citation_words * 1.5:
            warnings.append(
                "Response significantly longer than citation content - "
                "may contain elaboration beyond source material"
            )
            risk_score += 0.1
        
        # Check for proper nouns not in citations
        response_proper_nouns = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response_text))
        citation_proper_nouns = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', ' '.join(citation_texts)))
        
        new_proper_nouns = response_proper_nouns - citation_proper_nouns
        # Filter out common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'Section', 'Act', 'Rule'}
        new_proper_nouns = new_proper_nouns - common_words
        
        if new_proper_nouns and len(new_proper_nouns) > 3:
            warnings.append(
                f"Response introduces {len(new_proper_nouns)} proper nouns not in citations: "
                f"{list(new_proper_nouns)[:5]}"
            )
            risk_score += 0.15
        
        return min(risk_score, 1.0), warnings
    
    def _check_hedging(self, text: str) -> List[str]:
        """Check for appropriate hedging/qualification in statements"""
        warnings = []
        
        # Count hedging phrases
        hedging_phrases = [
            'may', 'might', 'could', 'possibly', 'likely', 'appears to',
            'seems to', 'suggests', 'indicates', 'according to', 'as per'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        factual_sentences = [s for s in sentences if len(s.strip()) > 20]
        
        hedged_sentences = 0
        for sentence in factual_sentences:
            if any(phrase in sentence.lower() for phrase in hedging_phrases):
                hedged_sentences += 1
        
        hedging_ratio = hedged_sentences / max(len(factual_sentences), 1)
        
        # Too little hedging may indicate overconfidence
        if hedging_ratio < 0.2:
            warnings.append(
                "Response contains few qualifying phrases - "
                "ensure claims are properly attributed to sources"
            )
        
        # Too much hedging may indicate uncertainty
        if hedging_ratio > 0.8:
            warnings.append(
                "Response heavily hedged - may indicate uncertainty about sources"
            )
        
        return warnings
    
    def validate_numeric_claim(
        self,
        claim_value: Any,
        citation_value: Any,
        tolerance: float = 0.01
    ) -> bool:
        """Validate that numeric claims match citations within tolerance"""
        try:
            claim_num = float(claim_value)
            citation_num = float(citation_value)
            
            # Check absolute difference
            diff = abs(claim_num - citation_num)
            relative_diff = diff / max(abs(citation_num), 1)
            
            return relative_diff <= tolerance
            
        except (ValueError, TypeError):
            # Not numeric, can't validate
            return True
    
    def validate_date_claim(
        self,
        claim_date: str,
        citation_date: str
    ) -> bool:
        """Validate that date claims match citations"""
        from dateutil import parser
        
        try:
            claim_dt = parser.parse(claim_date, fuzzy=True)
            citation_dt = parser.parse(citation_date, fuzzy=True)
            
            # Dates must match exactly
            return claim_dt.date() == citation_dt.date()
            
        except (ValueError, TypeError):
            # Can't parse, do string comparison
            return claim_date.lower().strip() == citation_date.lower().strip()
    
    def generate_validation_report(
        self,
        validation_result: Dict[str, Any]
    ) -> str:
        """Generate human-readable validation report"""
        report = []
        
        report.append("=" * 60)
        report.append("CITATION VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        status = "✅ VALID" if validation_result['valid'] else "❌ INVALID"
        report.append(f"\nStatus: {status}")
        report.append(f"Confidence: {validation_result['confidence']:.1%}")
        report.append(f"Hallucination Risk: {validation_result['hallucination_risk']:.1%}")
        
        # Citation coverage
        report.append(f"\nCitation Coverage:")
        report.append(f"  Total Claims: {validation_result['total_claims']}")
        report.append(f"  Cited Claims: {validation_result['total_claims'] - len(validation_result['uncited_claims'])}")
        report.append(f"  Uncited Claims: {len(validation_result['uncited_claims'])}")
        report.append(f"  Coverage: {validation_result['citation_coverage']:.1%}")
        
        # Errors
        if validation_result['errors']:
            report.append(f"\n❌ Errors ({len(validation_result['errors'])}):")
            for error in validation_result['errors']:
                report.append(f"  - {error}")
        
        # Warnings
        if validation_result['warnings']:
            report.append(f"\n⚠️  Warnings ({len(validation_result['warnings'])}):")
            for warning in validation_result['warnings']:
                report.append(f"  - {warning}")
        
        # Uncited claims
        if validation_result['uncited_claims']:
            report.append(f"\n📝 Uncited Claims:")
            for claim in validation_result['uncited_claims'][:5]:  # Show first 5
                report.append(f"  - {claim[:100]}...")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test validation
    import asyncio
    
    async def test():
        validator = CitationValidator(min_citations=2, similarity_threshold=0.6)
        
        # Mock response and citations
        response = """
        The dropout rate for SC students in 2016-17 was 15.2% in Anantapur district.
        This represents an increase from the previous year.
        The government has allocated 500 crores for infrastructure improvement under Nadu Nedu.
        """
        
        # Mock citations
        from dataclasses import dataclass
        
        @dataclass
        class MockCitation:
            excerpt: str
            doc_type: str
        
        citations = [
            MockCitation(
                excerpt="SC student dropout rate was 15.2% in Anantapur district for academic year 2016-17",
                doc_type="statistics"
            ),
            MockCitation(
                excerpt="Nadu Nedu scheme allocation is 500 crores for infrastructure development",
                doc_type="government_order"
            )
        ]
        
        # Validate
        result = await validator.validate(response, citations)
        
        # Print report
        print(validator.generate_validation_report(result))
    
    asyncio.run(test())