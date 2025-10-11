#!/usr/bin/env python3
"""
Pipeline Validation Utility
Validates the complete AP education policy intelligence pipeline
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# Neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)

class PipelineValidator:
    """Production-ready pipeline validator for AP education policy intelligence"""
    
    def __init__(self):
        # Database configurations
        self.pg_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'ap_education_policy',
            'user': 'postgres',
            'password': 'password'
        }
        
        self.neo4j_config = {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password'
        }
        
        # Validation thresholds
        self.thresholds = {
            'extraction_accuracy': 0.95,
            'normalization_coverage': 0.90,
            'fact_completeness': 0.95,
            'embedding_coverage': 0.90,
            'graph_completeness': 0.90,
            'query_latency': 3.0,  # seconds
            'data_consistency': 0.98
        }
        
        # Expected data ranges
        self.expected_ranges = {
            'GER': (0.5, 1.2),
            'NER': (0.4, 1.0),
            'GPI': (0.7, 1.3),
            'PTR': (10, 50),
            'Dropout_Rate': (0, 30),
            'Retention_Rate': (70, 100),
            'Enrolment': (1000, 1000000),
            'Teachers': (50, 50000),
            'Schools': (10, 10000),
            'Classrooms': (50, 50000)
        }
    
    def validate_extraction_stage(self) -> Dict[str, Any]:
        """Validate Stage 1: Table Extraction"""
        logger.info("Validating extraction stage...")
        
        validation_result = {
            'stage': 'extraction',
            'status': 'unknown',
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check extracted data file
            extracted_file = Path("data/extracted/all_extracted_data.json")
            if not extracted_file.exists():
                validation_result['issues'].append("Extracted data file not found")
                validation_result['status'] = 'failed'
                return validation_result
            
            # Load extracted data
            with open(extracted_file, 'r', encoding='utf-8') as f:
                extracted_data = json.load(f)
            
            # Calculate metrics
            total_pdfs = len(extracted_data)
            total_items = sum(len(items) for items in extracted_data.values())
            
            # Check extraction methods
            extraction_methods = {}
            for pdf_name, items in extracted_data.items():
                for item in items:
                    method = item.get('extraction_method', 'unknown')
                    extraction_methods[method] = extraction_methods.get(method, 0) + 1
            
            # Check confidence scores
            confidence_scores = []
            for pdf_name, items in extracted_data.items():
                for item in items:
                    if 'confidence' in item:
                        confidence_scores.append(item['confidence'])
            
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            # Update metrics
            validation_result['metrics'] = {
                'total_pdfs': total_pdfs,
                'total_items': total_items,
                'avg_confidence': avg_confidence,
                'extraction_methods': extraction_methods
            }
            
            # Validate against thresholds
            if avg_confidence >= self.thresholds['extraction_accuracy']:
                validation_result['status'] = 'passed'
            else:
                validation_result['status'] = 'warning'
                validation_result['issues'].append(f"Low extraction confidence: {avg_confidence:.3f}")
                validation_result['recommendations'].append("Consider improving OCR settings or table detection")
            
            if total_items == 0:
                validation_result['status'] = 'failed'
                validation_result['issues'].append("No items extracted")
            
        except Exception as e:
            validation_result['status'] = 'failed'
            validation_result['issues'].append(f"Validation error: {e}")
        
        return validation_result
    
    def validate_normalization_stage(self) -> Dict[str, Any]:
        """Validate Stage 2: Schema Normalization"""
        logger.info("Validating normalization stage...")
        
        validation_result = {
            'stage': 'normalization',
            'status': 'unknown',
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check normalized data file
            normalized_file = Path("data/normalized/normalized_facts.json")
            if not normalized_file.exists():
                validation_result['issues'].append("Normalized data file not found")
                validation_result['status'] = 'failed'
                return validation_result
            
            # Load normalized data
            with open(normalized_file, 'r', encoding='utf-8') as f:
                normalized_facts = json.load(f)
            
            if not normalized_facts:
                validation_result['issues'].append("No normalized facts found")
                validation_result['status'] = 'failed'
                return validation_result
            
            # Calculate metrics
            total_facts = len(normalized_facts)
            unique_indicators = len(set(fact['indicator'] for fact in normalized_facts))
            unique_districts = len(set(fact['district'] for fact in normalized_facts))
            unique_years = len(set(fact['year'] for fact in normalized_facts))
            
            # Check data quality
            unknown_indicators = sum(1 for fact in normalized_facts if fact['indicator'] == 'Unknown')
            unknown_districts = sum(1 for fact in normalized_facts if fact['district'] == 'Unknown')
            missing_values = sum(1 for fact in normalized_facts if fact['value'] is None)
            
            # Check value ranges
            out_of_range_count = 0
            for fact in normalized_facts:
                indicator = fact['indicator']
                value = fact['value']
                if indicator in self.expected_ranges and value is not None:
                    min_val, max_val = self.expected_ranges[indicator]
                    if not (min_val <= value <= max_val):
                        out_of_range_count += 1
            
            # Update metrics
            validation_result['metrics'] = {
                'total_facts': total_facts,
                'unique_indicators': unique_indicators,
                'unique_districts': unique_districts,
                'unique_years': unique_years,
                'unknown_indicators': unknown_indicators,
                'unknown_districts': unknown_districts,
                'missing_values': missing_values,
                'out_of_range_values': out_of_range_count
            }
            
            # Validate against thresholds
            if unknown_indicators / total_facts <= (1 - self.thresholds['normalization_coverage']):
                validation_result['status'] = 'passed'
            else:
                validation_result['status'] = 'warning'
                validation_result['issues'].append(f"High unknown indicators: {unknown_indicators/total_facts:.3f}")
                validation_result['recommendations'].append("Improve indicator normalization")
            
            if out_of_range_count > 0:
                validation_result['issues'].append(f"Out of range values: {out_of_range_count}")
                validation_result['recommendations'].append("Review data validation rules")
            
        except Exception as e:
            validation_result['status'] = 'failed'
            validation_result['issues'].append(f"Validation error: {e}")
        
        return validation_result
    
    def validate_fact_table_stage(self) -> Dict[str, Any]:
        """Validate Stage 3: Fact Table Building"""
        logger.info("Validating fact table stage...")
        
        validation_result = {
            'stage': 'fact_table',
            'status': 'unknown',
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Test PostgreSQL connection
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Check table existence
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('facts', 'documents', 'entities')
            """)
            
            tables = [row['table_name'] for row in cursor.fetchall()]
            
            if len(tables) != 3:
                validation_result['issues'].append(f"Missing tables: {3 - len(tables)}")
                validation_result['status'] = 'failed'
                return validation_result
            
            # Get fact statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_facts,
                    COUNT(embedding) as facts_with_embeddings,
                    COUNT(DISTINCT indicator) as unique_indicators,
                    COUNT(DISTINCT district) as unique_districts,
                    COUNT(DISTINCT year) as unique_years,
                    AVG(confidence) as avg_confidence
                FROM facts
            """)
            
            fact_stats = cursor.fetchone()
            
            # Get document statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(DISTINCT source_type) as unique_source_types
                FROM documents
            """)
            
            doc_stats = cursor.fetchone()
            
            # Get entity statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_entities,
                    COUNT(DISTINCT entity_type) as unique_entity_types
                FROM entities
            """)
            
            entity_stats = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            # Update metrics
            validation_result['metrics'] = {
                'facts': dict(fact_stats),
                'documents': dict(doc_stats),
                'entities': dict(entity_stats)
            }
            
            # Validate against thresholds
            embedding_coverage = fact_stats['facts_with_embeddings'] / fact_stats['total_facts'] if fact_stats['total_facts'] > 0 else 0
            
            if embedding_coverage >= self.thresholds['embedding_coverage']:
                validation_result['status'] = 'passed'
            else:
                validation_result['status'] = 'warning'
                validation_result['issues'].append(f"Low embedding coverage: {embedding_coverage:.3f}")
                validation_result['recommendations'].append("Generate missing embeddings")
            
            if fact_stats['total_facts'] == 0:
                validation_result['status'] = 'failed'
                validation_result['issues'].append("No facts in database")
            
        except Exception as e:
            validation_result['status'] = 'failed'
            validation_result['issues'].append(f"Validation error: {e}")
        
        return validation_result
    
    def validate_neo4j_stage(self) -> Dict[str, Any]:
        """Validate Stage 4: Neo4j Graph Loading"""
        logger.info("Validating Neo4j stage...")
        
        validation_result = {
            'stage': 'neo4j',
            'status': 'unknown',
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        if not NEO4J_AVAILABLE:
            validation_result['status'] = 'failed'
            validation_result['issues'].append("Neo4j driver not available")
            return validation_result
        
        try:
            # Test Neo4j connection
            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )
            
            with driver.session() as session:
                # Get node counts
                node_counts = {}
                node_types = ['Fact', 'Indicator', 'District', 'Year', 'Source']
                
                for node_type in node_types:
                    result = session.run(f"MATCH (n:{node_type}) RETURN COUNT(n) as count")
                    node_counts[node_type] = result.single()['count']
                
                # Get relationship counts
                rel_counts = {}
                rel_types = ['MEASURED_BY', 'LOCATED_IN', 'OBSERVED_IN', 'REPORTED_BY']
                
                for rel_type in rel_types:
                    result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) as count")
                    rel_counts[rel_type] = result.single()['count']
                
                # Check graph completeness
                total_nodes = sum(node_counts.values())
                total_relationships = sum(rel_counts.values())
                
                # Update metrics
                validation_result['metrics'] = {
                    'node_counts': node_counts,
                    'relationship_counts': rel_counts,
                    'total_nodes': total_nodes,
                    'total_relationships': total_relationships
                }
                
                # Validate against thresholds
                if total_nodes > 0 and total_relationships > 0:
                    validation_result['status'] = 'passed'
                else:
                    validation_result['status'] = 'failed'
                    validation_result['issues'].append("Empty graph")
                
                if node_counts['Fact'] == 0:
                    validation_result['status'] = 'failed'
                    validation_result['issues'].append("No facts in graph")
            
            driver.close()
            
        except Exception as e:
            validation_result['status'] = 'failed'
            validation_result['issues'].append(f"Validation error: {e}")
        
        return validation_result
    
    def validate_api_stage(self) -> Dict[str, Any]:
        """Validate Stage 6: RAG API"""
        logger.info("Validating API stage...")
        
        validation_result = {
            'stage': 'api',
            'status': 'unknown',
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            import requests
            
            # Test API endpoints
            api_base_url = "http://localhost:8000"
            
            # Health check
            try:
                response = requests.get(f"{api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    validation_result['metrics']['health_status'] = health_data.get('status', 'unknown')
                else:
                    validation_result['issues'].append("API health check failed")
            except:
                validation_result['issues'].append("API not accessible")
                validation_result['status'] = 'failed'
                return validation_result
            
            # Test search endpoint
            try:
                search_data = {
                    "query": "GER in Visakhapatnam",
                    "limit": 5,
                    "include_vector": True,
                    "include_graph": True
                }
                
                response = requests.post(f"{api_base_url}/search", json=search_data, timeout=10)
                if response.status_code == 200:
                    search_results = response.json()
                    validation_result['metrics']['search_results'] = search_results.get('total_count', 0)
                    validation_result['metrics']['query_time'] = search_results.get('query_time', 0)
                else:
                    validation_result['issues'].append("Search endpoint failed")
            except:
                validation_result['issues'].append("Search endpoint not accessible")
            
            # Test statistics endpoint
            try:
                response = requests.get(f"{api_base_url}/statistics", timeout=5)
                if response.status_code == 200:
                    stats_data = response.json()
                    validation_result['metrics']['api_stats'] = stats_data
                else:
                    validation_result['issues'].append("Statistics endpoint failed")
            except:
                validation_result['issues'].append("Statistics endpoint not accessible")
            
            # Validate against thresholds
            query_time = validation_result['metrics'].get('query_time', 0)
            if query_time <= self.thresholds['query_latency']:
                validation_result['status'] = 'passed'
            else:
                validation_result['status'] = 'warning'
                validation_result['issues'].append(f"Slow query response: {query_time:.3f}s")
                validation_result['recommendations'].append("Optimize query performance")
            
        except Exception as e:
            validation_result['status'] = 'failed'
            validation_result['issues'].append(f"Validation error: {e}")
        
        return validation_result
    
    def validate_data_consistency(self) -> Dict[str, Any]:
        """Validate data consistency across stages"""
        logger.info("Validating data consistency...")
        
        validation_result = {
            'stage': 'consistency',
            'status': 'unknown',
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Load normalized facts
            normalized_file = Path("data/normalized/normalized_facts.json")
            if not normalized_file.exists():
                validation_result['issues'].append("Normalized data file not found")
                validation_result['status'] = 'failed'
                return validation_result
            
            with open(normalized_file, 'r', encoding='utf-8') as f:
                normalized_facts = json.load(f)
            
            # Check PostgreSQL consistency
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT COUNT(*) as count FROM facts")
            pg_fact_count = cursor.fetchone()['count']
            
            cursor.close()
            conn.close()
            
            # Check Neo4j consistency
            neo4j_fact_count = 0
            if NEO4J_AVAILABLE:
                try:
                    driver = GraphDatabase.driver(
                        self.neo4j_config['uri'],
                        auth=(self.neo4j_config['user'], self.neo4j_config['password'])
                    )
                    
                    with driver.session() as session:
                        result = session.run("MATCH (f:Fact) RETURN COUNT(f) as count")
                        neo4j_fact_count = result.single()['count']
                    
                    driver.close()
                except:
                    pass
            
            # Calculate consistency metrics
            normalized_count = len(normalized_facts)
            pg_consistency = pg_fact_count / normalized_count if normalized_count > 0 else 0
            neo4j_consistency = neo4j_fact_count / normalized_count if normalized_count > 0 else 0
            
            # Update metrics
            validation_result['metrics'] = {
                'normalized_facts': normalized_count,
                'postgresql_facts': pg_fact_count,
                'neo4j_facts': neo4j_fact_count,
                'postgresql_consistency': pg_consistency,
                'neo4j_consistency': neo4j_consistency
            }
            
            # Validate against thresholds
            if pg_consistency >= self.thresholds['data_consistency']:
                validation_result['status'] = 'passed'
            else:
                validation_result['status'] = 'warning'
                validation_result['issues'].append(f"Low PostgreSQL consistency: {pg_consistency:.3f}")
                validation_result['recommendations'].append("Check PostgreSQL data loading")
            
            if neo4j_consistency < self.thresholds['data_consistency']:
                validation_result['issues'].append(f"Low Neo4j consistency: {neo4j_consistency:.3f}")
                validation_result['recommendations'].append("Check Neo4j data loading")
            
        except Exception as e:
            validation_result['status'] = 'failed'
            validation_result['issues'].append(f"Validation error: {e}")
        
        return validation_result
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete pipeline validation"""
        logger.info("Running full pipeline validation...")
        
        validation_results = {
            'validation_info': {
                'timestamp': datetime.now().isoformat(),
                'validator_version': '1.0.0',
                'thresholds': self.thresholds
            },
            'stages': {},
            'overall_status': 'unknown',
            'summary': {}
        }
        
        # Validate each stage
        stages = [
            'extraction',
            'normalization',
            'fact_table',
            'neo4j',
            'api',
            'consistency'
        ]
        
        for stage in stages:
            if stage == 'extraction':
                result = self.validate_extraction_stage()
            elif stage == 'normalization':
                result = self.validate_normalization_stage()
            elif stage == 'fact_table':
                result = self.validate_fact_table_stage()
            elif stage == 'neo4j':
                result = self.validate_neo4j_stage()
            elif stage == 'api':
                result = self.validate_api_stage()
            elif stage == 'consistency':
                result = self.validate_data_consistency()
            
            validation_results['stages'][stage] = result
        
        # Calculate overall status
        stage_statuses = [result['status'] for result in validation_results['stages'].values()]
        
        if all(status == 'passed' for status in stage_statuses):
            validation_results['overall_status'] = 'passed'
        elif any(status == 'failed' for status in stage_statuses):
            validation_results['overall_status'] = 'failed'
        else:
            validation_results['overall_status'] = 'warning'
        
        # Generate summary
        validation_results['summary'] = {
            'total_stages': len(stages),
            'passed_stages': sum(1 for status in stage_statuses if status == 'passed'),
            'warning_stages': sum(1 for status in stage_statuses if status == 'warning'),
            'failed_stages': sum(1 for status in stage_statuses if status == 'failed'),
            'overall_status': validation_results['overall_status']
        }
        
        return validation_results
    
    def save_validation_report(self, validation_results: Dict[str, Any]):
        """Save validation report to file"""
        report_file = Path("data/validation_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved to {report_file}")

def main():
    """Main function to run pipeline validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate AP Education Policy Intelligence Pipeline')
    parser.add_argument('--stage', help='Validate specific stage only')
    parser.add_argument('--output-file', default='data/validation_report.json',
                       help='Output file for validation report')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize validator
    validator = PipelineValidator()
    
    try:
        if args.stage:
            # Validate specific stage
            if args.stage == 'extraction':
                result = validator.validate_extraction_stage()
            elif args.stage == 'normalization':
                result = validator.validate_normalization_stage()
            elif args.stage == 'fact_table':
                result = validator.validate_fact_table_stage()
            elif args.stage == 'neo4j':
                result = validator.validate_neo4j_stage()
            elif args.stage == 'api':
                result = validator.validate_api_stage()
            elif args.stage == 'consistency':
                result = validator.validate_data_consistency()
            else:
                logger.error(f"Unknown stage: {args.stage}")
                sys.exit(1)
            
            print(json.dumps(result, indent=2))
        else:
            # Run full validation
            validation_results = validator.run_full_validation()
            validator.save_validation_report(validation_results)
            
            # Print summary
            summary = validation_results['summary']
            print("\n" + "="*60)
            print("PIPELINE VALIDATION SUMMARY")
            print("="*60)
            print(f"Overall Status: {summary['overall_status'].upper()}")
            print(f"Total Stages: {summary['total_stages']}")
            print(f"Passed: {summary['passed_stages']}")
            print(f"Warnings: {summary['warning_stages']}")
            print(f"Failed: {summary['failed_stages']}")
            print("="*60)
            
            # Print stage details
            for stage_name, stage_result in validation_results['stages'].items():
                status_icon = "✅" if stage_result['status'] == 'passed' else "⚠️" if stage_result['status'] == 'warning' else "❌"
                print(f"{status_icon} {stage_name}: {stage_result['status']}")
                
                if stage_result['issues']:
                    for issue in stage_result['issues']:
                        print(f"   - {issue}")
            
            print("="*60)
            print(f"Validation report saved to: {args.output_file}")
            
            sys.exit(0 if summary['overall_status'] == 'passed' else 1)
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
