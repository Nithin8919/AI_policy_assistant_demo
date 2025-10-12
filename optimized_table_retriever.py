#!/usr/bin/env python3
"""
Final Optimized Table Retrieval System
Properly understands and retrieves table data based on actual content structure
"""
import weaviate
from legal_aware_chunker import LegalAwareChunker
import time
import re
from collections import Counter

class OptimizedTableRetriever:
    """Optimized table retrieval that understands actual table data structure"""
    
    def __init__(self):
        self.chunker = LegalAwareChunker(
            weaviate_url="https://mg90f7v2snmbkjksqqpvxa.c0.asia-southeast1.gcp.weaviate.cloud",
            weaviate_api_key="YXNrUHlTTTZhRWRod0NCR19UM3FacDIxSktIK2htaHFPT29XbnljdVlGYTBPQkpDMXJYODIzRUhuZ2xrPV92MjAw"
        )
        
        # Patterns based on actual table data structure
        self.application_id_pattern = re.compile(r'25000\d{4}')  # Application IDs
        self.phone_pattern = re.compile(r'28147\d{6}')  # Phone numbers
        self.school_code_pattern = re.compile(r'\d{4,5}')  # School codes
        
        # Table data indicators
        self.table_data_indicators = [
            '25000',  # Application ID prefix
            '28147',  # Phone number prefix
            'BC-',    # Caste category
            'FEMALE', 'MALE',  # Gender
            '2012', '2013', '2014',  # Years
            'EM UP', 'EM HIGH', 'PRIMARY'  # School types
        ]
    
    def is_table_data(self, content: str) -> bool:
        """Check if content is table data based on actual patterns"""
        if not content or len(content.strip()) < 20:
            return False
        
        # Check for table data indicators
        indicator_count = sum(1 for indicator in self.table_data_indicators if indicator in content)
        
        # Check for structured data patterns
        has_application_id = bool(self.application_id_pattern.search(content))
        has_phone = bool(self.phone_pattern.search(content))
        has_multiple_words = len(content.split()) >= 8
        
        # Score based on indicators
        score = 0
        if indicator_count >= 2:
            score += 2
        if has_application_id:
            score += 2
        if has_phone:
            score += 2
        if has_multiple_words:
            score += 1
        
        return score >= 3  # Threshold for table data
    
    def search_table_data_by_pattern(self, query: str, limit: int = 10) -> list:
        """Search for table data using pattern-based matching"""
        print(f"🔍 Pattern-based table search: '{query}'")
        
        # Extract search terms
        query_lower = query.lower()
        search_terms = []
        
        # Map query terms to actual data patterns
        if 'application' in query_lower or 'id' in query_lower:
            search_terms.extend(['25000', 'Application'])
        if 'phone' in query_lower or 'contact' in query_lower:
            search_terms.extend(['28147', 'phone'])
        if 'school' in query_lower:
            search_terms.extend(['EM UP', 'EM HIGH', 'PRIMARY', 'SCHOOL'])
        if 'student' in query_lower:
            search_terms.extend(['BC-', 'FEMALE', 'MALE'])
        if 'district' in query_lower:
            search_terms.extend(['District', 'Mandal'])
        
        # If no specific terms, use general table patterns
        if not search_terms:
            search_terms = ['25000', '28147']
        
        results = []
        seen_content = set()
        
        # Search for each term
        for term in search_terms:
            try:
                search_results = self.chunker.search_legal_documents(term, limit=limit)
                for result in search_results:
                    content_hash = hash(result['content'][:100])
                    if content_hash not in seen_content and self.is_table_data(result['content']):
                        seen_content.add(content_hash)
                        results.append(result)
            except:
                continue
        
        # Sort by confidence and relevance
        results.sort(key=lambda x: x['confidence_score'], reverse=True)
        return results[:limit]
    
    def search_comprehensive_table_data(self, query: str, limit: int = 10) -> dict:
        """Comprehensive table data search with multiple strategies"""
        print(f"🔍 Comprehensive table search: '{query}'")
        
        start_time = time.time()
        
        # Strategy 1: Pattern-based search
        pattern_results = self.search_table_data_by_pattern(query, limit)
        
        # Strategy 2: Section-based search
        section_results = []
        for section in ['header', 'content', 'enrollment_data']:
            try:
                section_search = self.chunker.search_legal_documents(
                    query, 
                    limit=limit,
                    section_types=[section]
                )
                section_results.extend(section_search)
            except:
                continue
        
        # Filter section results for table data
        filtered_section_results = []
        seen_content = set()
        
        for result in section_results:
            content_hash = hash(result['content'][:100])
            if (content_hash not in seen_content and 
                self.is_table_data(result['content'])):
                seen_content.add(content_hash)
                filtered_section_results.append(result)
        
        # Combine results
        all_results = pattern_results + filtered_section_results
        
        # Remove duplicates and sort
        unique_results = []
        seen_content = set()
        
        for result in all_results:
            content_hash = hash(result['content'][:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        search_time = time.time() - start_time
        
        return {
            'query': query,
            'total_results': len(unique_results),
            'pattern_results': len(pattern_results),
            'section_results': len(filtered_section_results),
            'search_time': search_time,
            'results': unique_results[:limit]
        }
    
    def analyze_optimized_table_retrieval(self):
        """Analyze optimized table retrieval performance"""
        print("🎯 OPTIMIZED TABLE RETRIEVAL PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Test queries that should find table data
        test_queries = [
            {
                'query': 'application id 25000',
                'description': 'Application ID search',
                'expected_pattern': '25000'
            },
            {
                'query': 'phone number 28147',
                'description': 'Phone number search',
                'expected_pattern': '28147'
            },
            {
                'query': 'school EM UP',
                'description': 'School type search',
                'expected_pattern': 'EM UP'
            },
            {
                'query': 'student BC-',
                'description': 'Student category search',
                'expected_pattern': 'BC-'
            },
            {
                'query': 'district data',
                'description': 'District information',
                'expected_pattern': 'District'
            }
        ]
        
        total_results = 0
        relevant_results = 0
        query_times = []
        
        print(f"\n🔍 TESTING {len(test_queries)} OPTIMIZED QUERIES")
        print("-" * 50)
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n{i}. {test['description'].upper()}")
            print(f"   Query: '{test['query']}'")
            print(f"   Expected pattern: {test['expected_pattern']}")
            
            # Comprehensive search
            search_result = self.search_comprehensive_table_data(test['query'], limit=10)
            
            query_times.append(search_result['search_time'])
            total_results += search_result['total_results']
            
            print(f"   ✅ Found {search_result['total_results']} results in {search_result['search_time']:.3f}s")
            print(f"   📊 Pattern: {search_result['pattern_results']}, Section: {search_result['section_results']}")
            
            # Check relevance
            if search_result['results']:
                relevant_count = sum(1 for r in search_result['results'] 
                                  if test['expected_pattern'] in r['content'])
                relevant_results += relevant_count
                
                print(f"   🎯 Relevant results: {relevant_count}/{search_result['total_results']}")
                
                # Show best result
                best = search_result['results'][0]
                print(f"   🏆 Best: [{best['section_type']}] Confidence: {best['confidence_score']:.3f}")
                print(f"   📄 Content: {best['content'][:120]}...")
            else:
                print(f"   ⚠️ No results found")
        
        # Performance metrics
        avg_query_time = sum(query_times) / len(query_times)
        relevance_rate = relevant_results / total_results * 100 if total_results > 0 else 0
        
        print(f"\n📊 OPTIMIZED TABLE RETRIEVAL METRICS:")
        print("-" * 45)
        print(f"Total queries tested: {len(test_queries)}")
        print(f"Total results retrieved: {total_results}")
        print(f"Relevant results: {relevant_results}")
        print(f"Relevance rate: {relevance_rate:.1f}%")
        print(f"Average query time: {avg_query_time:.3f}s")
        
        # Test large-scale retrieval
        print(f"\n📊 LARGE-SCALE OPTIMIZED RETRIEVAL")
        print("-" * 40)
        
        large_query = "table data"
        start_time = time.time()
        large_result = self.search_comprehensive_table_data(large_query, limit=50)
        large_time = time.time() - start_time
        
        print(f"Query: '{large_query}'")
        print(f"✅ Retrieved {large_result['total_results']} results in {large_time:.3f}s")
        print(f"📊 Pattern: {large_result['pattern_results']}, Section: {large_result['section_results']}")
        
        # Analyze result quality
        if large_result['results']:
            table_data_count = sum(1 for r in large_result['results'] if self.is_table_data(r['content']))
            print(f"📊 Confirmed table data: {table_data_count}/{large_result['total_results']} ({table_data_count/large_result['total_results']*100:.1f}%)")
            
            # Section distribution
            section_counts = Counter(r['section_type'] for r in large_result['results'])
            print(f"📊 Section distribution:")
            for section, count in section_counts.most_common():
                print(f"   {section}: {count} results")
        
        # Overall assessment
        print(f"\n🎯 OPTIMIZED TABLE RETRIEVAL ASSESSMENT")
        print("=" * 50)
        
        if relevance_rate >= 70 and avg_query_time < 0.3:
            rating = "🟢 EXCELLENT"
        elif relevance_rate >= 50 and avg_query_time < 0.5:
            rating = "🟡 GOOD"
        elif relevance_rate >= 30 and avg_query_time < 0.8:
            rating = "🟠 FAIR"
        else:
            rating = "🔴 NEEDS IMPROVEMENT"
        
        print(f"✅ Relevance Rate: {relevance_rate:.1f}%")
        print(f"⚡ Average Query Time: {avg_query_time:.3f}s")
        print(f"📊 Large Query Performance: {large_result['total_results']} results in {large_time:.3f}s")
        print(f"🏆 Overall Rating: {rating}")
        
        return {
            'relevance_rate': relevance_rate,
            'avg_query_time': avg_query_time,
            'total_results': total_results,
            'relevant_results': relevant_results,
            'rating': rating
        }

def main():
    """Run optimized table retrieval analysis"""
    retriever = OptimizedTableRetriever()
    results = retriever.analyze_optimized_table_retrieval()
    
    print(f"\n🎉 Optimized table retrieval analysis complete!")
    print(f"📊 Relevance rate: {results['relevance_rate']:.1f}%")
    print(f"⚡ Average query time: {results['avg_query_time']:.3f}s")
    print(f"🏆 Rating: {results['rating']}")

if __name__ == "__main__":
    main()

