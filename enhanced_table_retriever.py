#!/usr/bin/env python3
"""
Enhanced Table Retrieval System
Improved retrieval for both table headers and data rows
"""
import weaviate
from legal_aware_chunker import LegalAwareChunker
import time
from collections import Counter

class EnhancedTableRetriever:
    """Enhanced table retrieval with better understanding of table structure"""
    
    def __init__(self):
        self.chunker = LegalAwareChunker(
            weaviate_url="https://mg90f7v2snmbkjksqqpvxa.c0.asia-southeast1.gcp.weaviate.cloud",
            weaviate_api_key="YXNrUHlTTTZhRWRod0NCR19UM3FacDIxSktIK2htaHFPT29XbnljdVlGYTBPQkpDMXJYODIzRUhuZ2xrPV92MjAw"
        )
        
        # Table patterns for better retrieval
        self.table_header_patterns = [
            'SL.No', 'District Name', 'Application Id', 'School Code', 
            'Mandal Name', 'Applicant Name', 'Parent/Guardian'
        ]
        
        self.table_data_patterns = [
            r'\d+\s+\w+\s+\w+',  # Number followed by words
            r'\d{4,}',  # Long numbers (IDs)
            r'\w+\s+\d{4}',  # Word followed by year
            r'\d+\.\d+',  # Decimal numbers
        ]
    
    def search_table_headers(self, query: str, limit: int = 10) -> list:
        """Search specifically for table headers"""
        print(f"🔍 Searching table headers: '{query}'")
        
        # Search for content containing table header patterns
        results = []
        for pattern in self.table_header_patterns:
            if pattern.lower() in query.lower():
                # Direct search for this pattern
                search_results = self.chunker.search_legal_documents(pattern, limit=limit)
                results.extend(search_results)
        
        # Remove duplicates and sort by confidence
        unique_results = []
        seen_content = set()
        
        for result in results:
            content_hash = hash(result['content'][:100])  # Use first 100 chars as identifier
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Sort by confidence
        unique_results.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return unique_results[:limit]
    
    def search_table_data(self, query: str, limit: int = 10) -> list:
        """Search for table data rows (not just headers)"""
        print(f"🔍 Searching table data: '{query}'")
        
        # Extract keywords from query
        keywords = query.lower().split()
        
        # Search for content that contains table data patterns
        results = []
        
        # Search by section type first (data is often in 'header' or 'content' sections)
        for section_type in ['header', 'content', 'enrollment_data']:
            try:
                section_results = self.chunker.search_legal_documents(
                    query, 
                    limit=limit,
                    section_types=[section_type]
                )
                results.extend(section_results)
            except:
                continue
        
        # Filter for actual table data (not just headers)
        table_data_results = []
        for result in results:
            content = result['content']
            
            # Check if it's table data (contains numbers and structured data)
            has_numbers = any(char.isdigit() for char in content)
            has_structure = len(content.split()) > 5  # Multiple words
            is_not_header = not any(pattern in content for pattern in self.table_header_patterns)
            
            if has_numbers and has_structure and is_not_header:
                table_data_results.append(result)
        
        return table_data_results[:limit]
    
    def search_comprehensive_tables(self, query: str, limit: int = 10) -> dict:
        """Comprehensive table search including both headers and data"""
        print(f"🔍 Comprehensive table search: '{query}'")
        
        start_time = time.time()
        
        # Search for headers
        header_results = self.search_table_headers(query, limit)
        
        # Search for data
        data_results = self.search_table_data(query, limit)
        
        # Combine and deduplicate
        all_results = header_results + data_results
        unique_results = []
        seen_content = set()
        
        for result in all_results:
            content_hash = hash(result['content'][:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Sort by confidence
        unique_results.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        search_time = time.time() - start_time
        
        return {
            'query': query,
            'total_results': len(unique_results),
            'header_results': len(header_results),
            'data_results': len(data_results),
            'search_time': search_time,
            'results': unique_results[:limit]
        }
    
    def analyze_table_retrieval_performance(self):
        """Analyze table retrieval performance comprehensively"""
        print("🎯 ENHANCED TABLE RETRIEVAL PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Test queries for different types of table content
        test_queries = [
            {
                'query': 'school enrollment data',
                'type': 'enrollment',
                'expected_patterns': ['SL.No', 'District', 'School']
            },
            {
                'query': 'application information',
                'type': 'application',
                'expected_patterns': ['Application Id', 'Applicant Name']
            },
            {
                'query': 'district statistics',
                'type': 'statistics',
                'expected_patterns': ['District Name', 'Mandal Name']
            },
            {
                'query': 'school codes',
                'type': 'school',
                'expected_patterns': ['School Code', 'School Name']
            },
            {
                'query': 'student data',
                'type': 'student',
                'expected_patterns': ['Student', 'Roll No', 'Class']
            }
        ]
        
        total_results = 0
        total_table_content = 0
        query_times = []
        
        print(f"\n🔍 TESTING {len(test_queries)} TABLE QUERY SCENARIOS")
        print("-" * 50)
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n{i}. {test['type'].upper()} QUERY")
            print(f"   Query: '{test['query']}'")
            print(f"   Expected patterns: {test['expected_patterns']}")
            
            # Comprehensive search
            search_result = self.search_comprehensive_tables(test['query'], limit=10)
            
            query_times.append(search_result['search_time'])
            total_results += search_result['total_results']
            
            print(f"   ✅ Found {search_result['total_results']} results in {search_result['search_time']:.3f}s")
            print(f"   📊 Headers: {search_result['header_results']}, Data: {search_result['data_results']}")
            
            # Analyze result quality
            if search_result['results']:
                table_content_count = 0
                for result in search_result['results']:
                    content = result['content']
                    if any(pattern in content for pattern in test['expected_patterns']):
                        table_content_count += 1
                
                total_table_content += table_content_count
                print(f"   🎯 Relevant table content: {table_content_count}/{search_result['total_results']}")
                
                # Show best result
                best = search_result['results'][0]
                print(f"   🏆 Best: [{best['section_type']}] Confidence: {best['confidence_score']:.3f}")
                print(f"   📄 Content: {best['content'][:100]}...")
            else:
                print(f"   ⚠️ No results found")
        
        # Performance metrics
        avg_query_time = sum(query_times) / len(query_times)
        table_retrieval_rate = total_table_content / total_results * 100 if total_results > 0 else 0
        
        print(f"\n📊 ENHANCED TABLE RETRIEVAL METRICS:")
        print("-" * 40)
        print(f"Total queries tested: {len(test_queries)}")
        print(f"Total results retrieved: {total_results}")
        print(f"Relevant table content: {total_table_content}")
        print(f"Table retrieval rate: {table_retrieval_rate:.1f}%")
        print(f"Average query time: {avg_query_time:.3f}s")
        
        # Test large-scale retrieval
        print(f"\n📊 LARGE-SCALE TABLE RETRIEVAL TEST")
        print("-" * 40)
        
        large_query = "school data"
        start_time = time.time()
        large_result = self.search_comprehensive_tables(large_query, limit=50)
        large_time = time.time() - start_time
        
        print(f"Query: '{large_query}'")
        print(f"✅ Retrieved {large_result['total_results']} results in {large_time:.3f}s")
        print(f"📊 Headers: {large_result['header_results']}, Data: {large_result['data_results']}")
        
        # Analyze section type distribution
        section_counts = Counter(r['section_type'] for r in large_result['results'])
        print(f"📊 Section distribution:")
        for section, count in section_counts.most_common():
            print(f"   {section}: {count} results")
        
        # Overall assessment
        print(f"\n🎯 ENHANCED TABLE RETRIEVAL ASSESSMENT")
        print("=" * 50)
        
        if table_retrieval_rate >= 60 and avg_query_time < 0.2:
            rating = "🟢 EXCELLENT"
        elif table_retrieval_rate >= 40 and avg_query_time < 0.3:
            rating = "🟡 GOOD"
        elif table_retrieval_rate >= 20 and avg_query_time < 0.5:
            rating = "🟠 FAIR"
        else:
            rating = "🔴 NEEDS IMPROVEMENT"
        
        print(f"✅ Table Retrieval Rate: {table_retrieval_rate:.1f}%")
        print(f"⚡ Average Query Time: {avg_query_time:.3f}s")
        print(f"📊 Large Query Performance: {large_result['total_results']} results in {large_time:.3f}s")
        print(f"🏆 Overall Rating: {rating}")
        
        return {
            'table_retrieval_rate': table_retrieval_rate,
            'avg_query_time': avg_query_time,
            'total_results': total_results,
            'table_content': total_table_content,
            'rating': rating
        }

def main():
    """Run enhanced table retrieval analysis"""
    retriever = EnhancedTableRetriever()
    results = retriever.analyze_table_retrieval_performance()
    
    print(f"\n🎉 Enhanced table retrieval analysis complete!")
    print(f"📊 Table retrieval rate: {results['table_retrieval_rate']:.1f}%")
    print(f"⚡ Average query time: {results['avg_query_time']:.3f}s")
    print(f"🏆 Rating: {results['rating']}")

if __name__ == "__main__":
    main()
