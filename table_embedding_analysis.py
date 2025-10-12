#!/usr/bin/env python3
"""
Table Embedding and Retrieval Analysis
Test how well table content is embedded and retrieved from Weaviate
"""
import weaviate
from legal_aware_chunker import LegalAwareChunker
import time
from collections import Counter

def analyze_table_embeddings_and_retrieval():
    """Analyze table embedding and retrieval performance"""
    
    print("🔍 TABLE EMBEDDING & RETRIEVAL ANALYSIS")
    print("=" * 60)
    
    # Initialize chunker
    chunker = LegalAwareChunker(
        weaviate_url="https://mg90f7v2snmbkjksqqpvxa.c0.asia-southeast1.gcp.weaviate.cloud",
        weaviate_api_key="YXNrUHlTTTZhRWRod0NCR19UM3FacDIxSktIK2htaHFPT29XbnljdVlGYTBPQkpDMXJYODIzRUhuZ2xrPV92MjAw"
    )
    
    print("✅ Connected to Weaviate successfully!")
    
    # Test table-specific queries
    table_queries = [
        "school enrollment data",
        "district wise statistics", 
        "application id numbers",
        "student roll numbers",
        "teacher recruitment data",
        "budget allocation tables",
        "performance metrics data",
        "administrative procedures table"
    ]
    
    print(f"\n🔍 TESTING TABLE-SPECIFIC QUERIES")
    print("-" * 40)
    
    total_results = 0
    table_results = 0
    query_times = []
    
    for i, query in enumerate(table_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        start_time = time.time()
        results = chunker.search_legal_documents(query, limit=10)
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        if results:
            print(f"   ✅ Found {len(results)} results in {query_time:.3f}s")
            total_results += len(results)
            
            # Analyze if results contain table content
            table_content_count = 0
            for result in results:
                content = result['content']
                # Check for table indicators
                if any(indicator in content for indicator in ['SL.No', 'District', 'Application Id', 'School Code']):
                    table_content_count += 1
            
            table_results += table_content_count
            print(f"   📊 Table content: {table_content_count}/{len(results)} results")
            
            # Show best result
            if results:
                best = results[0]
                print(f"   🏆 Best: [{best['section_type']}] Confidence: {best['confidence_score']:.3f}")
                print(f"   📄 Content: {best['content'][:100]}...")
        else:
            print(f"   ⚠️ No results found in {query_time:.3f}s")
    
    # Calculate table retrieval efficiency
    avg_query_time = sum(query_times) / len(query_times)
    table_retrieval_rate = table_results / total_results * 100 if total_results > 0 else 0
    
    print(f"\n📊 TABLE RETRIEVAL PERFORMANCE:")
    print("-" * 35)
    print(f"Total queries: {len(table_queries)}")
    print(f"Total results: {total_results}")
    print(f"Table content results: {table_results}")
    print(f"Table retrieval rate: {table_retrieval_rate:.1f}%")
    print(f"Average query time: {avg_query_time:.3f}s")
    
    # Test specific table data retrieval
    print(f"\n🎯 SPECIFIC TABLE DATA RETRIEVAL TESTS")
    print("-" * 45)
    
    specific_tests = [
        {
            "query": "SL.No District Name",
            "description": "Table headers"
        },
        {
            "query": "Application Id Applicant Name",
            "description": "Application data"
        },
        {
            "query": "School Code School Name",
            "description": "School information"
        },
        {
            "query": "Parent Guardian Name",
            "description": "Parent/Guardian data"
        }
    ]
    
    specific_results = []
    
    for test in specific_tests:
        print(f"\n🔍 {test['description']}")
        print(f"   Query: '{test['query']}'")
        
        start_time = time.time()
        results = chunker.search_legal_documents(test['query'], limit=5)
        query_time = time.time() - start_time
        
        if results:
            print(f"   ✅ Found {len(results)} results in {query_time:.3f}s")
            
            # Check result quality
            high_quality = sum(1 for r in results if r['confidence_score'] >= 0.8)
            print(f"   📊 High quality results: {high_quality}/{len(results)}")
            
            # Show sample
            if results:
                sample = results[0]
                print(f"   📄 Sample: {sample['content'][:80]}...")
            
            specific_results.append({
                'query': test['query'],
                'count': len(results),
                'time': query_time,
                'high_quality': high_quality
            })
        else:
            print(f"   ⚠️ No results found")
            specific_results.append({
                'query': test['query'],
                'count': 0,
                'time': query_time,
                'high_quality': 0
            })
    
    # Test filtering by section type for tables
    print(f"\n🎯 SECTION-TYPE FILTERING FOR TABLES")
    print("-" * 40)
    
    section_tests = [
        {"section": "enrollment_data", "query": "enrollment statistics"},
        {"section": "budget_data", "query": "budget allocation"},
        {"section": "content", "query": "school information"},
        {"section": "header", "query": "table headers"}
    ]
    
    section_results = []
    
    for test in section_tests:
        print(f"\n🔍 {test['section']} section")
        print(f"   Query: '{test['query']}'")
        
        start_time = time.time()
        results = chunker.search_legal_documents(
            test['query'], 
            limit=5,
            section_types=[test['section']]
        )
        query_time = time.time() - start_time
        
        if results:
            print(f"   ✅ Found {len(results)} results in {query_time:.3f}s")
            avg_conf = sum(r['confidence_score'] for r in results) / len(results)
            print(f"   📊 Average confidence: {avg_conf:.3f}")
            
            section_results.append({
                'section': test['section'],
                'count': len(results),
                'time': query_time,
                'avg_confidence': avg_conf
            })
        else:
            print(f"   ⚠️ No results found")
            section_results.append({
                'section': test['section'],
                'count': 0,
                'time': query_time,
                'avg_confidence': 0
            })
    
    # Test large table data retrieval
    print(f"\n📊 LARGE TABLE DATA RETRIEVAL TEST")
    print("-" * 40)
    
    print("🔍 Testing large result set for table content...")
    start_time = time.time()
    large_results = chunker.search_legal_documents("school data", limit=100)
    large_time = time.time() - start_time
    
    if large_results:
        print(f"✅ Retrieved {len(large_results)} results in {large_time:.3f}s")
        
        # Analyze table content in large set
        table_content = sum(1 for r in large_results if any(indicator in r['content'] for indicator in ['SL.No', 'District', 'Application Id']))
        print(f"📊 Table content: {table_content}/{len(large_results)} results")
        
        # Analyze confidence distribution
        confidences = [r['confidence_score'] for r in large_results]
        avg_conf = sum(confidences) / len(confidences)
        high_conf = sum(1 for c in confidences if c >= 0.8)
        
        print(f"📈 Average confidence: {avg_conf:.3f}")
        print(f"📈 High confidence results: {high_conf}/{len(large_results)} ({high_conf/len(large_results)*100:.1f}%)")
        
        # Analyze section type distribution
        section_counts = Counter(r['section_type'] for r in large_results)
        print(f"📊 Section distribution:")
        for section, count in section_counts.most_common():
            print(f"   {section}: {count} results")
    
    # Overall assessment
    print(f"\n🎯 TABLE EMBEDDING & RETRIEVAL ASSESSMENT")
    print("=" * 50)
    
    # Calculate overall metrics
    total_specific_results = sum(r['count'] for r in specific_results)
    total_section_results = sum(r['count'] for r in section_results)
    avg_specific_time = sum(r['time'] for r in specific_results) / len(specific_results)
    avg_section_time = sum(r['time'] for r in section_results) / len(section_results)
    
    print(f"✅ Table Query Success Rate: {len([r for r in specific_results if r['count'] > 0])}/{len(specific_results)} ({len([r for r in specific_results if r['count'] > 0])/len(specific_results)*100:.1f}%)")
    print(f"⚡ Average Query Time: {avg_query_time:.3f}s")
    print(f"📊 Table Retrieval Rate: {table_retrieval_rate:.1f}%")
    print(f"🎯 Specific Table Queries: {total_specific_results} results")
    print(f"🔍 Section Filtering: {total_section_results} results")
    
    # Performance rating
    if table_retrieval_rate >= 70 and avg_query_time < 0.1:
        rating = "🟢 EXCELLENT"
    elif table_retrieval_rate >= 50 and avg_query_time < 0.2:
        rating = "🟡 GOOD"
    elif table_retrieval_rate >= 30 and avg_query_time < 0.5:
        rating = "🟠 FAIR"
    else:
        rating = "🔴 NEEDS IMPROVEMENT"
    
    print(f"\n🏆 Overall Table Retrieval Rating: {rating}")
    
    return {
        'table_retrieval_rate': table_retrieval_rate,
        'avg_query_time': avg_query_time,
        'total_results': total_results,
        'table_results': table_results,
        'rating': rating
    }

if __name__ == "__main__":
    results = analyze_table_embeddings_and_retrieval()
    print(f"\n🎉 Table embedding and retrieval analysis complete!")
    print(f"📊 Table retrieval rate: {results['table_retrieval_rate']:.1f}%")
    print(f"⚡ Average query time: {results['avg_query_time']:.3f}s")
    print(f"🏆 Rating: {results['rating']}")

