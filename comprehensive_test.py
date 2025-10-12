#!/usr/bin/env python3
"""
Comprehensive Weaviate Legal Document System Test
Tests efficiency, accuracy, and functionality of the legal document search system
"""
import time
import weaviate
from legal_aware_chunker import LegalAwareChunker
import json
from typing import List, Dict, Any

class WeaviateSystemTester:
    """Comprehensive tester for the Weaviate legal document system"""
    
    def __init__(self):
        self.weaviate_url = "https://mg90f7v2snmbkjksqqpvxa.c0.asia-southeast1.gcp.weaviate.cloud"
        self.api_key = "YXNrUHlTTTZhRWRod0NCR19UM3FacDIxSktIK2htaHFPT29XbnljdVlGYTBPQkpDMXJYODIzRUhuZ2xrPV92MjAw"
        self.collection_name = "LegalDocuments"
        
        # Initialize chunker
        self.chunker = LegalAwareChunker(
            weaviate_url=self.weaviate_url,
            weaviate_api_key=self.api_key
        )
        
        self.test_results = {}
    
    def test_connection(self) -> bool:
        """Test Weaviate connection"""
        print("🔌 Testing Weaviate Connection...")
        start_time = time.time()
        
        try:
            if not self.chunker.weaviate_client:
                print("❌ Failed to connect to Weaviate")
                return False
            
            # Test basic query
            result = self.chunker.weaviate_client.query.aggregate(self.collection_name).with_meta_count().do()
            count = result['data']['Aggregate'][self.collection_name][0]['meta']['count']
            
            connection_time = time.time() - start_time
            print(f"✅ Connected successfully in {connection_time:.3f}s")
            print(f"📊 Collection contains {count} objects")
            
            self.test_results['connection'] = {
                'success': True,
                'time': connection_time,
                'object_count': count
            }
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.test_results['connection'] = {'success': False, 'error': str(e)}
            return False
    
    def test_basic_search(self) -> Dict[str, Any]:
        """Test basic search functionality"""
        print("\n🔍 Testing Basic Search Functionality...")
        
        test_queries = [
            "school enrollment",
            "budget allocation",
            "teacher recruitment",
            "education policy",
            "district administration",
            "student performance",
            "government orders",
            "SCERT guidelines"
        ]
        
        results = {}
        
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            start_time = time.time()
            
            try:
                # Test the chunker's search method
                search_results = self.chunker.search_legal_documents(
                    query, 
                    limit=5,
                    min_confidence=0.3
                )
                
                search_time = time.time() - start_time
                
                if search_results:
                    print(f"   ✅ Found {len(search_results)} results in {search_time:.3f}s")
                    
                    # Show top result
                    top_result = search_results[0]
                    print(f"   🏆 Top result: [{top_result['section_type']}] Confidence: {top_result['confidence_score']:.3f}")
                    print(f"   📄 Content: {top_result['content'][:100]}...")
                    
                    results[query] = {
                        'success': True,
                        'time': search_time,
                        'result_count': len(search_results),
                        'top_confidence': top_result['confidence_score'],
                        'top_section_type': top_result['section_type']
                    }
                else:
                    print(f"   ⚠️ No results found in {search_time:.3f}s")
                    results[query] = {
                        'success': False,
                        'time': search_time,
                        'result_count': 0
                    }
                    
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
                results[query] = {
                    'success': False,
                    'error': str(e)
                }
        
        self.test_results['basic_search'] = results
        return results
    
    def test_filtered_search(self) -> Dict[str, Any]:
        """Test filtered search by section type and confidence"""
        print("\n🎯 Testing Filtered Search...")
        
        filters = [
            {'section_types': ['content'], 'min_confidence': 0.8, 'name': 'High-confidence content'},
            {'section_types': ['enrollment_data'], 'min_confidence': 0.7, 'name': 'Enrollment data'},
            {'section_types': ['budget_data'], 'min_confidence': 0.6, 'name': 'Budget information'},
            {'section_types': ['procedure'], 'min_confidence': 0.9, 'name': 'Procedures'},
            {'section_types': ['definition'], 'min_confidence': 0.8, 'name': 'Definitions'}
        ]
        
        results = {}
        
        for filter_config in filters:
            print(f"\n🔍 Filter: {filter_config['name']}")
            start_time = time.time()
            
            try:
                search_results = self.chunker.search_legal_documents(
                    "education policy",
                    limit=3,
                    section_types=filter_config['section_types'],
                    min_confidence=filter_config['min_confidence']
                )
                
                search_time = time.time() - start_time
                
                if search_results:
                    print(f"   ✅ Found {len(search_results)} results in {search_time:.3f}s")
                    avg_confidence = sum(r['confidence_score'] for r in search_results) / len(search_results)
                    print(f"   📊 Average confidence: {avg_confidence:.3f}")
                    
                    results[filter_config['name']] = {
                        'success': True,
                        'time': search_time,
                        'result_count': len(search_results),
                        'avg_confidence': avg_confidence
                    }
                else:
                    print(f"   ⚠️ No results found in {search_time:.3f}s")
                    results[filter_config['name']] = {
                        'success': False,
                        'time': search_time,
                        'result_count': 0
                    }
                    
            except Exception as e:
                print(f"   ❌ Filtered search failed: {e}")
                results[filter_config['name']] = {
                    'success': False,
                    'error': str(e)
                }
        
        self.test_results['filtered_search'] = results
        return results
    
    def test_performance_metrics(self) -> Dict[str, Any]:
        """Test system performance metrics"""
        print("\n⚡ Testing Performance Metrics...")
        
        # Test 1: Response time for different query lengths
        print("\n1. Testing response time by query complexity:")
        query_tests = [
            ("short", "school"),
            ("medium", "school enrollment policy"),
            ("long", "school enrollment policy and budget allocation for education")
        ]
        
        performance_results = {}
        
        for test_type, query in query_tests:
            times = []
            for i in range(3):  # Run 3 times for average
                start_time = time.time()
                try:
                    results = self.chunker.search_legal_documents(query, limit=5)
                    times.append(time.time() - start_time)
                except:
                    times.append(float('inf'))
            
            avg_time = sum(times) / len(times) if times else 0
            print(f"   {test_type.capitalize()} query: {avg_time:.3f}s average")
            performance_results[f'{test_type}_query_time'] = avg_time
        
        # Test 2: Batch query performance
        print("\n2. Testing batch query performance:")
        batch_queries = [
            "school enrollment", "budget allocation", "teacher recruitment",
            "education policy", "district administration", "student performance"
        ]
        
        start_time = time.time()
        batch_results = []
        for query in batch_queries:
            try:
                results = self.chunker.search_legal_documents(query, limit=3)
                batch_results.append(len(results))
            except:
                batch_results.append(0)
        
        batch_time = time.time() - start_time
        print(f"   Batch of {len(batch_queries)} queries: {batch_time:.3f}s")
        print(f"   Average per query: {batch_time/len(batch_queries):.3f}s")
        
        performance_results['batch_query_time'] = batch_time
        performance_results['avg_per_query'] = batch_time / len(batch_queries)
        
        # Test 3: Large result set performance
        print("\n3. Testing large result set performance:")
        start_time = time.time()
        try:
            large_results = self.chunker.search_legal_documents("education", limit=50)
            large_time = time.time() - start_time
            print(f"   Large result set (50 items): {large_time:.3f}s")
            print(f"   Retrieved {len(large_results)} results")
            performance_results['large_result_time'] = large_time
            performance_results['large_result_count'] = len(large_results)
        except Exception as e:
            print(f"   ❌ Large result test failed: {e}")
            performance_results['large_result_time'] = float('inf')
        
        self.test_results['performance'] = performance_results
        return performance_results
    
    def test_data_quality(self) -> Dict[str, Any]:
        """Test data quality and accuracy"""
        print("\n📊 Testing Data Quality...")
        
        # Test 1: Confidence score distribution
        print("\n1. Testing confidence score distribution:")
        try:
            # Get a sample of results
            sample_results = self.chunker.search_legal_documents("education", limit=100)
            
            if sample_results:
                confidences = [r['confidence_score'] for r in sample_results]
                avg_confidence = sum(confidences) / len(confidences)
                min_confidence = min(confidences)
                max_confidence = max(confidences)
                
                print(f"   Average confidence: {avg_confidence:.3f}")
                print(f"   Min confidence: {min_confidence:.3f}")
                print(f"   Max confidence: {max_confidence:.3f}")
                
                # Count by confidence ranges
                high_conf = sum(1 for c in confidences if c >= 0.8)
                med_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
                low_conf = sum(1 for c in confidences if c < 0.5)
                
                print(f"   High confidence (≥0.8): {high_conf} ({high_conf/len(confidences)*100:.1f}%)")
                print(f"   Medium confidence (0.5-0.8): {med_conf} ({med_conf/len(confidences)*100:.1f}%)")
                print(f"   Low confidence (<0.5): {low_conf} ({low_conf/len(confidences)*100:.1f}%)")
                
                quality_results = {
                    'avg_confidence': avg_confidence,
                    'min_confidence': min_confidence,
                    'max_confidence': max_confidence,
                    'high_confidence_count': high_conf,
                    'high_confidence_percentage': high_conf/len(confidences)*100,
                    'total_samples': len(confidences)
                }
            else:
                print("   ❌ No sample results available")
                quality_results = {'error': 'No sample results'}
                
        except Exception as e:
            print(f"   ❌ Confidence analysis failed: {e}")
            quality_results = {'error': str(e)}
        
        # Test 2: Section type distribution
        print("\n2. Testing section type distribution:")
        try:
            section_counts = {}
            for section_type in ['header', 'content', 'enrollment_data', 'budget_data', 'procedure', 'definition']:
                results = self.chunker.search_legal_documents(
                    "education", 
                    limit=100,
                    section_types=[section_type]
                )
                section_counts[section_type] = len(results)
                print(f"   {section_type}: {len(results)} results")
            
            quality_results['section_distribution'] = section_counts
            
        except Exception as e:
            print(f"   ❌ Section type analysis failed: {e}")
            quality_results['section_error'] = str(e)
        
        self.test_results['data_quality'] = quality_results
        return quality_results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling"""
        print("\n🧪 Testing Edge Cases...")
        
        edge_cases = [
            ("empty_query", ""),
            ("very_long_query", "a" * 1000),
            ("special_chars", "!@#$%^&*()"),
            ("numbers_only", "123456789"),
            ("unicode", "école éducation"),
            ("sql_injection", "'; DROP TABLE users; --"),
            ("very_specific", "very specific technical term that probably doesn't exist")
        ]
        
        edge_results = {}
        
        for case_name, query in edge_cases:
            print(f"\n🔍 Testing: {case_name}")
            start_time = time.time()
            
            try:
                results = self.chunker.search_legal_documents(query, limit=5)
                search_time = time.time() - start_time
                
                print(f"   ✅ Handled gracefully in {search_time:.3f}s")
                print(f"   📊 Returned {len(results)} results")
                
                edge_results[case_name] = {
                    'success': True,
                    'time': search_time,
                    'result_count': len(results)
                }
                
            except Exception as e:
                search_time = time.time() - start_time
                print(f"   ❌ Failed: {e}")
                edge_results[case_name] = {
                    'success': False,
                    'time': search_time,
                    'error': str(e)
                }
        
        self.test_results['edge_cases'] = edge_results
        return edge_results
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        print("\n📋 Generating Comprehensive Test Report...")
        
        report = []
        report.append("=" * 80)
        report.append("WEAVIATE LEGAL DOCUMENT SYSTEM - COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        
        # Connection test
        if 'connection' in self.test_results:
            conn = self.test_results['connection']
            report.append(f"\n🔌 CONNECTION TEST:")
            report.append(f"   Status: {'✅ SUCCESS' if conn['success'] else '❌ FAILED'}")
            if conn['success']:
                report.append(f"   Response Time: {conn['time']:.3f}s")
                report.append(f"   Object Count: {conn['object_count']:,}")
            else:
                report.append(f"   Error: {conn['error']}")
        
        # Basic search test
        if 'basic_search' in self.test_results:
            report.append(f"\n🔍 BASIC SEARCH TEST:")
            successful_searches = sum(1 for r in self.test_results['basic_search'].values() if r['success'])
            total_searches = len(self.test_results['basic_search'])
            report.append(f"   Success Rate: {successful_searches}/{total_searches} ({successful_searches/total_searches*100:.1f}%)")
            
            if successful_searches > 0:
                avg_time = sum(r['time'] for r in self.test_results['basic_search'].values() if r['success']) / successful_searches
                report.append(f"   Average Response Time: {avg_time:.3f}s")
                
                # Best performing query
                best_query = max(
                    [(k, v) for k, v in self.test_results['basic_search'].items() if v['success']],
                    key=lambda x: x[1]['top_confidence']
                )
                report.append(f"   Best Query: '{best_query[0]}' (confidence: {best_query[1]['top_confidence']:.3f})")
        
        # Performance test
        if 'performance' in self.test_results:
            perf = self.test_results['performance']
            report.append(f"\n⚡ PERFORMANCE METRICS:")
            report.append(f"   Short Query Time: {perf.get('short_query_time', 0):.3f}s")
            report.append(f"   Medium Query Time: {perf.get('medium_query_time', 0):.3f}s")
            report.append(f"   Long Query Time: {perf.get('long_query_time', 0):.3f}s")
            report.append(f"   Batch Query Time: {perf.get('batch_query_time', 0):.3f}s")
            report.append(f"   Average Per Query: {perf.get('avg_per_query', 0):.3f}s")
        
        # Data quality test
        if 'data_quality' in self.test_results:
            quality = self.test_results['data_quality']
            report.append(f"\n📊 DATA QUALITY:")
            if 'avg_confidence' in quality:
                report.append(f"   Average Confidence: {quality['avg_confidence']:.3f}")
                report.append(f"   High Confidence Rate: {quality.get('high_confidence_percentage', 0):.1f}%")
                report.append(f"   Sample Size: {quality.get('total_samples', 0)}")
        
        # Edge cases test
        if 'edge_cases' in self.test_results:
            edge = self.test_results['edge_cases']
            successful_edge = sum(1 for r in edge.values() if r['success'])
            total_edge = len(edge)
            report.append(f"\n🧪 EDGE CASE HANDLING:")
            report.append(f"   Success Rate: {successful_edge}/{total_edge} ({successful_edge/total_edge*100:.1f}%)")
        
        # Overall assessment
        report.append(f"\n🎯 OVERALL ASSESSMENT:")
        
        # Calculate overall score
        scores = []
        if 'connection' in self.test_results and self.test_results['connection']['success']:
            scores.append(100)
        
        if 'basic_search' in self.test_results:
            search_success_rate = sum(1 for r in self.test_results['basic_search'].values() if r['success']) / len(self.test_results['basic_search'])
            scores.append(search_success_rate * 100)
        
        if 'performance' in self.test_results:
            # Score based on response time (lower is better)
            avg_time = self.test_results['performance'].get('avg_per_query', 1)
            perf_score = max(0, 100 - (avg_time * 100))  # Penalize slow responses
            scores.append(perf_score)
        
        if scores:
            overall_score = sum(scores) / len(scores)
            report.append(f"   Overall Score: {overall_score:.1f}/100")
            
            if overall_score >= 90:
                report.append(f"   Status: 🟢 EXCELLENT")
            elif overall_score >= 75:
                report.append(f"   Status: 🟡 GOOD")
            elif overall_score >= 60:
                report.append(f"   Status: 🟠 FAIR")
            else:
                report.append(f"   Status: 🔴 NEEDS IMPROVEMENT")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🚀 Starting Comprehensive Weaviate System Test")
        print("=" * 60)
        
        # Run all tests
        self.test_connection()
        self.test_basic_search()
        self.test_filtered_search()
        self.test_performance_metrics()
        self.test_data_quality()
        self.test_edge_cases()
        
        # Generate and display report
        report = self.generate_report()
        print(report)
        
        # Save report to file
        with open('weaviate_test_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: weaviate_test_report.txt")
        
        return self.test_results

def main():
    """Run comprehensive test suite"""
    tester = WeaviateSystemTester()
    results = tester.run_all_tests()
    
    print(f"\n🎉 Test suite completed!")
    print(f"📊 Test results available in 'results' variable")

if __name__ == "__main__":
    main()

