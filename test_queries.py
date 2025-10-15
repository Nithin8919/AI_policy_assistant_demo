#!/usr/bin/env python3
"""
Test Query Bank for AP Policy Co-Pilot
30 comprehensive test queries covering legal, data, and policy implementation
"""
import asyncio
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TestQuery:
    """Test query with expected answer type"""
    id: str
    query: str
    category: str  # 'legal', 'data', 'combined', 'temporal', 'comparative'
    difficulty: str  # 'easy', 'medium', 'hard'
    expected_sources: List[str]  # Expected document types
    expected_citations_min: int  # Minimum expected citations
    notes: str = ""

class TestQueryBank:
    """Comprehensive test query bank"""
    
    def __init__(self):
        self.queries = self._initialize_queries()
        logger.info(f"Initialized {len(self.queries)} test queries")
    
    def _initialize_queries(self) -> List[TestQuery]:
        """Initialize all test queries"""
        queries = []
        
        # ===== LEGAL QUERIES (Pure law/policy lookup) =====
        
        queries.append(TestQuery(
            id="L001",
            query="What are the responsibilities of School Management Committees under AP law?",
            category="legal",
            difficulty="easy",
            expected_sources=["act", "rule"],
            expected_citations_min=2,
            notes="Should cite AP Education Act 1982 and AP SMC Act 1998"
        ))
        
        queries.append(TestQuery(
            id="L002",
            query="Under which rule can private school fees be regulated in Andhra Pradesh?",
            category="legal",
            difficulty="medium",
            expected_sources=["act", "rule", "go"],
            expected_citations_min=2,
            notes="Should reference RTE Act 2009 and AP Private Schools Regulation"
        ))
        
        queries.append(TestQuery(
            id="L003",
            query="Which Government Order governs the Nadu-Nedu infrastructure improvement scheme?",
            category="legal",
            difficulty="easy",
            expected_sources=["go"],
            expected_citations_min=1,
            notes="Should identify specific GO number and date"
        ))
        
        queries.append(TestQuery(
            id="L004",
            query="What is the legal framework for implementing Right to Education in AP?",
            category="legal",
            difficulty="medium",
            expected_sources=["central_act", "state_act", "rule"],
            expected_citations_min=3,
            notes="Should show hierarchy: RTE Act 2009 → AP RTE Rules 2010 → Implementation GOs"
        ))
        
        queries.append(TestQuery(
            id="L005",
            query="What are the eligibility criteria for teachers as per NCTE regulations?",
            category="legal",
            difficulty="medium",
            expected_sources=["rule", "framework"],
            expected_citations_min=2,
            notes="Should cite NCTE regulations and AP teacher recruitment rules"
        ))
        
        # ===== DATA QUERIES (Pure statistics lookup) =====
        
        queries.append(TestQuery(
            id="D001",
            query="What was the dropout rate among ST students in Andhra Pradesh in 2016-17?",
            category="data",
            difficulty="easy",
            expected_sources=["statistics"],
            expected_citations_min=1,
            notes="Should provide specific percentage with source"
        ))
        
        queries.append(TestQuery(
            id="D002",
            query="How many schools in Andhra Pradesh offered Telugu as medium of instruction in 2016-17?",
            category="data",
            difficulty="easy",
            expected_sources=["statistics"],
            expected_citations_min=1,
            notes="Should provide count and source (UDISE+ or Educational Statistics)"
        ))
        
        queries.append(TestQuery(
            id="D003",
            query="What is the Pupil-Teacher Ratio in government schools across all AP districts?",
            category="data",
            difficulty="medium",
            expected_sources=["statistics"],
            expected_citations_min=13,  # One per district
            notes="Should provide district-wise breakdown"
        ))
        
        queries.append(TestQuery(
            id="D004",
            query="How much budget was allocated for school education in AP in 2022-23?",
            category="data",
            difficulty="easy",
            expected_sources=["budget"],
            expected_citations_min=1,
            notes="Should cite budget document with specific allocation"
        ))
        
        queries.append(TestQuery(
            id="D005",
            query="What percentage of schools in Visakhapatnam district have computer labs?",
            category="data",
            difficulty="medium",
            expected_sources=["statistics"],
            expected_citations_min=1,
            notes="Should provide specific percentage for Visakhapatnam"
        ))
        
        # ===== COMBINED QUERIES (Law + Data) =====
        
        queries.append(TestQuery(
            id="C001",
            query="What is the legal framework for mid-day meals, and what was the actual budget spent on it in 2021-22?",
            category="combined",
            difficulty="medium",
            expected_sources=["go", "budget"],
            expected_citations_min=2,
            notes="Should combine Jagananna Gorumudda GO with budget expenditure data"
        ))
        
        queries.append(TestQuery(
            id="C002",
            query="According to RTE Act, what should be the PTR, and what is the actual PTR in government schools?",
            category="combined",
            difficulty="medium",
            expected_sources=["central_act", "statistics"],
            expected_citations_min=2,
            notes="Should compare legal requirement with actual data"
        ))
        
        queries.append(TestQuery(
            id="C003",
            query="What are the infrastructure norms under Nadu-Nedu, and how many schools have met these norms?",
            category="combined",
            difficulty="hard",
            expected_sources=["go", "statistics"],
            expected_citations_min=3,
            notes="Should link GO specifications with implementation data"
        ))
        
        queries.append(TestQuery(
            id="C004",
            query="What incentives does the Amma Vodi scheme provide, and how many students benefited in 2020-21?",
            category="combined",
            difficulty="medium",
            expected_sources=["go", "statistics"],
            expected_citations_min=2,
            notes="Should combine scheme details with beneficiary data"
        ))
        
        queries.append(TestQuery(
            id="C005",
            query="What is the legal requirement for SC/ST enrollment, and what are the actual enrollment numbers?",
            category="combined",
            difficulty="medium",
            expected_sources=["central_act", "statistics"],
            expected_citations_min=2,
            notes="Should combine constitutional/legal provisions with UDISE+ data"
        ))
        
        # ===== TEMPORAL QUERIES (Trends over time) =====
        
        queries.append(TestQuery(
            id="T001",
            query="How has the dropout rate changed in Anantapur district from 2015 to 2020?",
            category="temporal",
            difficulty="medium",
            expected_sources=["statistics"],
            expected_citations_min=3,  # Multiple years
            notes="Should show year-by-year trend"
        ))
        
        queries.append(TestQuery(
            id="T002",
            query="What is the trend in education budget allocation over the last 5 years?",
            category="temporal",
            difficulty="medium",
            expected_sources=["budget"],
            expected_citations_min=5,
            notes="Should provide multi-year budget data"
        ))
        
        queries.append(TestQuery(
            id="T003",
            query="How has girl enrollment in primary schools evolved from 2010 to 2020?",
            category="temporal",
            difficulty="medium",
            expected_sources=["statistics"],
            expected_citations_min=5,
            notes="Should show decade-long trend"
        ))
        
        queries.append(TestQuery(
            id="T004",
            query="Compare teacher recruitment numbers before and after 2019?",
            category="temporal",
            difficulty="hard",
            expected_sources=["statistics", "go"],
            expected_citations_min=3,
            notes="Should identify policy changes affecting recruitment"
        ))
        
        # ===== COMPARATIVE QUERIES (Cross-district/category) =====
        
        queries.append(TestQuery(
            id="COM001",
            query="Compare dropout rates between coastal and Rayalaseema districts",
            category="comparative",
            difficulty="hard",
            expected_sources=["statistics"],
            expected_citations_min=6,  # Multiple districts
            notes="Should group districts and compare"
        ))
        
        queries.append(TestQuery(
            id="COM002",
            query="Which district has the highest enrollment of SC students?",
            category="comparative",
            difficulty="easy",
            expected_sources=["statistics"],
            expected_citations_min=1,
            notes="Should rank districts"
        ))
        
        queries.append(TestQuery(
            id="COM003",
            query="Compare infrastructure availability between government and private schools",
            category="comparative",
            difficulty="hard",
            expected_sources=["statistics"],
            expected_citations_min=2,
            notes="Should compare management types"
        ))
        
        queries.append(TestQuery(
            id="COM004",
            query="Which districts have the best and worst PTR?",
            category="comparative",
            difficulty="medium",
            expected_sources=["statistics"],
            expected_citations_min=2,
            notes="Should identify extremes"
        ))
        
        # ===== COMPLEX/ANALYTICAL QUERIES =====
        
        queries.append(TestQuery(
            id="A001",
            query="What is the correlation between budget allocation and dropout rates across districts?",
            category="combined",
            difficulty="hard",
            expected_sources=["budget", "statistics"],
            expected_citations_min=10,
            notes="Requires cross-dataset analysis"
        ))
        
        queries.append(TestQuery(
            id="A002",
            query="Identify gaps between RTE norms and actual school infrastructure in tribal areas",
            category="combined",
            difficulty="hard",
            expected_sources=["central_act", "statistics"],
            expected_citations_min=5,
            notes="Requires comparing legal standards with data"
        ))
        
        queries.append(TestQuery(
            id="A003",
            query="What schemes target girl education, and what has been their impact on enrollment?",
            category="combined",
            difficulty="hard",
            expected_sources=["go", "statistics"],
            expected_citations_min=4,
            notes="Should link multiple schemes with outcome data"
        ))
        
        queries.append(TestQuery(
            id="A004",
            query="How does teacher training policy align with actual training completion rates?",
            category="combined",
            difficulty="hard",
            expected_sources=["rule", "statistics"],
            expected_citations_min=3,
            notes="Policy vs implementation analysis"
        ))
        
        queries.append(TestQuery(
            id="A005",
            query="What is the implementation status of NEP 2020 recommendations in AP?",
            category="combined",
            difficulty="hard",
            expected_sources=["framework", "go", "statistics"],
            expected_citations_min=5,
            notes="Requires tracking policy to implementation"
        ))
        
        # Additional queries to reach 30
        
        queries.append(TestQuery(
            id="L006",
            query="What is the composition and tenure of District Education Committees?",
            category="legal",
            difficulty="medium",
            expected_sources=["rule"],
            expected_citations_min=1,
            notes="Should cite relevant rules"
        ))
        
        return queries
    
    def get_query(self, query_id: str) -> TestQuery:
        """Get query by ID"""
        for query in self.queries:
            if query.id == query_id:
                return query
        raise ValueError(f"Query {query_id} not found")
    
    def get_queries_by_category(self, category: str) -> List[TestQuery]:
        """Get all queries in a category"""
        return [q for q in self.queries if q.category == category]
    
    def get_queries_by_difficulty(self, difficulty: str) -> List[TestQuery]:
        """Get all queries of a difficulty level"""
        return [q for q in self.queries if q.difficulty == difficulty]
    
    async def run_test_suite(
        self,
        orchestrator,
        output_file: str = "test_results.json"
    ):
        """Run all test queries through the orchestrator"""
        logger.info(f"🧪 Running test suite with {len(self.queries)} queries...")
        
        results = []
        
        for i, test_query in enumerate(self.queries, 1):
            logger.info(f"\n[{i}/{len(self.queries)}] Testing {test_query.id}: {test_query.query[:50]}...")
            
            try:
                # Run query
                start_time = datetime.now()
                response = await orchestrator.query(test_query.query)
                end_time = datetime.now()
                
                # Evaluate result
                passed, issues = self._evaluate_response(test_query, response)
                
                result = {
                    'query_id': test_query.id,
                    'query': test_query.query,
                    'category': test_query.category,
                    'difficulty': test_query.difficulty,
                    'passed': passed,
                    'issues': issues,
                    'num_citations': len(response.citations),
                    'confidence': response.confidence_score,
                    'processing_time': (end_time - start_time).total_seconds(),
                    'response_length': len(response.answer),
                    'legal_chain_length': len(response.legal_chain),
                    'warnings': response.warnings
                }
                
                results.append(result)
                
                status = "✅ PASS" if passed else "❌ FAIL"
                logger.info(f"   {status} ({len(response.citations)} citations, {response.confidence_score:.2%} confidence)")
                
                if issues:
                    logger.warning(f"   Issues: {', '.join(issues)}")
                
            except Exception as e:
                logger.error(f"   ❌ ERROR: {e}")
                results.append({
                    'query_id': test_query.id,
                    'query': test_query.query,
                    'passed': False,
                    'issues': [f"Exception: {str(e)}"],
                    'error': str(e)
                })
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(self.queries),
            'summary': summary,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\n📊 Test Results Summary:")
        logger.info(f"   Total: {summary['total']}")
        logger.info(f"   Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
        logger.info(f"   Failed: {summary['failed']}")
        logger.info(f"   Avg Confidence: {summary['avg_confidence']:.2%}")
        logger.info(f"   Avg Citations: {summary['avg_citations']:.1f}")
        logger.info(f"   Avg Time: {summary['avg_time']:.2f}s")
        logger.info(f"\n✅ Results saved to {output_file}")
        
        return results, summary
    
    def _evaluate_response(
        self,
        test_query: TestQuery,
        response
    ) -> tuple[bool, List[str]]:
        """Evaluate if response meets test query expectations"""
        issues = []
        
        # Check minimum citations
        if len(response.citations) < test_query.expected_citations_min:
            issues.append(
                f"Insufficient citations: {len(response.citations)} < {test_query.expected_citations_min}"
            )
        
        # Check expected sources
        citation_types = set()
        for citation in response.citations:
            if hasattr(citation, 'doc_type'):
                doc_type = str(citation.doc_type).lower()
                for expected in test_query.expected_sources:
                    if expected in doc_type:
                        citation_types.add(expected)
        
        missing_sources = set(test_query.expected_sources) - citation_types
        if missing_sources:
            issues.append(f"Missing expected sources: {missing_sources}")
        
        # Check confidence
        if response.confidence_score < 0.6:
            issues.append(f"Low confidence: {response.confidence_score:.2%}")
        
        # Check for warnings
        if response.warnings:
            issues.append(f"Validation warnings: {len(response.warnings)}")
        
        passed = len(issues) == 0
        return passed, issues
    
    def _calculate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        total = len(results)
        passed = sum(1 for r in results if r.get('passed', False))
        
        valid_results = [r for r in results if 'confidence' in r]
        
        return {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0,
            'avg_confidence': sum(r['confidence'] for r in valid_results) / len(valid_results) if valid_results else 0,
            'avg_citations': sum(r['num_citations'] for r in valid_results) / len(valid_results) if valid_results else 0,
            'avg_time': sum(r['processing_time'] for r in valid_results) / len(valid_results) if valid_results else 0,
            'by_category': self._stats_by_category(valid_results),
            'by_difficulty': self._stats_by_difficulty(valid_results)
        }
    
    def _stats_by_category(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate stats by category"""
        categories = {}
        for r in results:
            cat = r.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0}
            categories[cat]['total'] += 1
            if r.get('passed'):
                categories[cat]['passed'] += 1
        
        for cat in categories:
            total = categories[cat]['total']
            passed = categories[cat]['passed']
            categories[cat]['pass_rate'] = passed / total if total > 0 else 0
        
        return categories
    
    def _stats_by_difficulty(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate stats by difficulty"""
        difficulties = {}
        for r in results:
            diff = r.get('difficulty', 'unknown')
            if diff not in difficulties:
                difficulties[diff] = {'total': 0, 'passed': 0}
            difficulties[diff]['total'] += 1
            if r.get('passed'):
                difficulties[diff]['passed'] += 1
        
        for diff in difficulties:
            total = difficulties[diff]['total']
            passed = difficulties[diff]['passed']
            difficulties[diff]['pass_rate'] = passed / total if total > 0 else 0
        
        return difficulties

if __name__ == "__main__":
    # Display all test queries
    bank = TestQueryBank()
    
    print("\n" + "=" * 80)
    print("AP POLICY CO-PILOT TEST QUERY BANK")
    print("=" * 80)
    
    for category in ['legal', 'data', 'combined', 'temporal', 'comparative']:
        queries = bank.get_queries_by_category(category)
        print(f"\n### {category.upper()} QUERIES ({len(queries)})")
        for q in queries:
            print(f"\n{q.id} [{q.difficulty}]")
            print(f"Q: {q.query}")
            print(f"Expected: {q.expected_citations_min}+ citations from {q.expected_sources}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {len(bank.queries)} test queries")
    print("=" * 80)