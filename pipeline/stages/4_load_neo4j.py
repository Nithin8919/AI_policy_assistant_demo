#!/usr/bin/env python3
"""
Stage 4: Neo4j Knowledge Graph Loader
Loads normalized facts into Neo4j with proper ontology and relationships
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

# Neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)

class Neo4jGraphLoader:
    """Production-ready Neo4j knowledge graph loader for AP education policy"""
    
    def __init__(self, output_dir: str = "data/neo4j"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Neo4j configuration
        self.neo4j_config = {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password'
        }
        
        # Initialize driver
        self.driver = None
        if NEO4J_AVAILABLE:
            try:
                self.driver = GraphDatabase.driver(
                    self.neo4j_config['uri'],
                    auth=(self.neo4j_config['user'], self.neo4j_config['password'])
                )
                logger.info("Neo4j driver initialized")
            except Exception as e:
                logger.error(f"Neo4j connection failed: {e}")
                self.driver = None
        else:
            logger.warning("Neo4j driver not available")
    
    def setup_graph_schema(self) -> bool:
        """Setup Neo4j graph schema with constraints and indexes"""
        if not self.driver:
            logger.error("Neo4j driver not available")
            return False
        
        try:
            with self.driver.session() as session:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE",
                    "CREATE CONSTRAINT indicator_name_unique IF NOT EXISTS FOR (i:Indicator) REQUIRE i.name IS UNIQUE",
                    "CREATE CONSTRAINT district_name_unique IF NOT EXISTS FOR (d:District) REQUIRE d.name IS UNIQUE",
                    "CREATE CONSTRAINT year_value_unique IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE",
                    "CREATE CONSTRAINT source_name_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.warning(f"Constraint creation failed: {e}")
                
                # Create indexes
                indexes = [
                    "CREATE INDEX fact_indicator IF NOT EXISTS FOR (f:Fact) ON (f.indicator)",
                    "CREATE INDEX fact_district IF NOT EXISTS FOR (f:Fact) ON (f.district)",
                    "CREATE INDEX fact_year IF NOT EXISTS FOR (f:Fact) ON (f.year)",
                    "CREATE INDEX fact_value IF NOT EXISTS FOR (f:Fact) ON (f.value)",
                    "CREATE INDEX fact_source IF NOT EXISTS FOR (f:Fact) ON (f.source)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.warning(f"Index creation failed: {e}")
                
                logger.info("Graph schema setup completed")
                return True
        
        except Exception as e:
            logger.error(f"Graph schema setup failed: {e}")
            return False
    
    def load_facts_to_graph(self, normalized_facts: List[Dict[str, Any]]) -> bool:
        """
        Load normalized facts into Neo4j graph
        
        Args:
            normalized_facts: List of normalized facts from Stage 2
            
        Returns:
            Success status
        """
        if not self.driver:
            logger.error("Neo4j driver not available")
            return False
        
        logger.info(f"Loading {len(normalized_facts)} facts to Neo4j")
        
        try:
            with self.driver.session() as session:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                
                # Load facts in batches
                batch_size = 1000
                total_inserted = 0
                
                for i in range(0, len(normalized_facts), batch_size):
                    batch = normalized_facts[i:i + batch_size]
                    inserted_count = self._load_fact_batch(session, batch)
                    total_inserted += inserted_count
                    
                    logger.info(f"Loaded batch {i//batch_size + 1}: {inserted_count} facts")
                
                logger.info(f"Successfully loaded {total_inserted} facts to Neo4j")
                return True
        
        except Exception as e:
            logger.error(f"Fact loading failed: {e}")
            return False
    
    def _load_fact_batch(self, session, facts: List[Dict[str, Any]]) -> int:
        """Load a batch of facts to Neo4j"""
        inserted_count = 0
        
        for fact in facts:
            try:
                # Create fact node
                fact_query = """
                CREATE (f:Fact {
                    fact_id: $fact_id,
                    indicator: $indicator,
                    category: $category,
                    district: $district,
                    year: $year,
                    value: $value,
                    unit: $unit,
                    source: $source,
                    page_ref: $page_ref,
                    confidence: $confidence,
                    table_id: $table_id,
                    pdf_name: $pdf_name,
                    created_at: $created_at
                })
                """
                
                session.run(fact_query, {
                    'fact_id': fact['fact_id'],
                    'indicator': fact['indicator'],
                    'category': fact.get('category', 'total'),
                    'district': fact['district'],
                    'year': fact['year'],
                    'value': float(fact['value']),
                    'unit': fact.get('unit', 'unknown'),
                    'source': fact['source'],
                    'page_ref': fact.get('page_ref', 0),
                    'confidence': fact.get('confidence', 0.8),
                    'table_id': fact.get('table_id', ''),
                    'pdf_name': fact.get('pdf_name', ''),
                    'created_at': datetime.now().isoformat()
                })
                
                # Create indicator node and relationship
                indicator_query = """
                MERGE (i:Indicator {name: $indicator})
                SET i.category = $category,
                    i.unit = $unit,
                    i.updated_at = $updated_at
                MERGE (f:Fact {fact_id: $fact_id})
                MERGE (f)-[:MEASURED_BY]->(i)
                """
                
                session.run(indicator_query, {
                    'indicator': fact['indicator'],
                    'category': fact.get('category', 'total'),
                    'unit': fact.get('unit', 'unknown'),
                    'fact_id': fact['fact_id'],
                    'updated_at': datetime.now().isoformat()
                })
                
                # Create district node and relationship
                district_query = """
                MERGE (d:District {name: $district})
                SET d.state = 'Andhra Pradesh',
                    d.updated_at = $updated_at
                MERGE (f:Fact {fact_id: $fact_id})
                MERGE (f)-[:LOCATED_IN]->(d)
                """
                
                session.run(district_query, {
                    'district': fact['district'],
                    'fact_id': fact['fact_id'],
                    'updated_at': datetime.now().isoformat()
                })
                
                # Create year node and relationship
                year_query = """
                MERGE (y:Year {value: $year})
                SET y.updated_at = $updated_at
                MERGE (f:Fact {fact_id: $fact_id})
                MERGE (f)-[:OBSERVED_IN]->(y)
                """
                
                session.run(year_query, {
                    'year': fact['year'],
                    'fact_id': fact['fact_id'],
                    'updated_at': datetime.now().isoformat()
                })
                
                # Create source node and relationship
                source_query = """
                MERGE (s:Source {name: $source})
                SET s.type = $source_type,
                    s.updated_at = $updated_at
                MERGE (f:Fact {fact_id: $fact_id})
                MERGE (f)-[:REPORTED_BY]->(s)
                """
                
                session.run(source_query, {
                    'source': fact['source'],
                    'source_type': self._classify_source_type(fact['source']),
                    'fact_id': fact['fact_id'],
                    'updated_at': datetime.now().isoformat()
                })
                
                inserted_count += 1
            
            except Exception as e:
                logger.error(f"Failed to load fact {fact.get('fact_id', 'unknown')}: {e}")
                continue
        
        return inserted_count
    
    def _classify_source_type(self, source: str) -> str:
        """Classify source type"""
        source_lower = source.lower()
        
        if 'cse' in source_lower:
            return 'Administrative'
        elif 'scert' in source_lower:
            return 'Academic'
        elif 'go' in source_lower:
            return 'Legal'
        else:
            return 'Unknown'
    
    def create_relationships(self) -> bool:
        """Create additional relationships between entities"""
        if not self.driver:
            logger.error("Neo4j driver not available")
            return False
        
        try:
            with self.driver.session() as session:
                # Create indicator relationships
                indicator_relationships = [
                    """
                    MATCH (i1:Indicator), (i2:Indicator)
                    WHERE i1.category = i2.category AND i1.name <> i2.name
                    MERGE (i1)-[:RELATED_TO]->(i2)
                    """,
                    
                    """
                    MATCH (i:Indicator)
                    WHERE i.category = 'enrolment'
                    SET i:EnrolmentIndicator
                    """,
                    
                    """
                    MATCH (i:Indicator)
                    WHERE i.category = 'teacher'
                    SET i:TeacherIndicator
                    """,
                    
                    """
                    MATCH (i:Indicator)
                    WHERE i.category = 'infrastructure'
                    SET i:InfrastructureIndicator
                    """
                ]
                
                for query in indicator_relationships:
                    try:
                        session.run(query)
                    except Exception as e:
                        logger.warning(f"Relationship creation failed: {e}")
                
                # Create district relationships
                district_relationships = [
                    """
                    MATCH (d:District)
                    WHERE d.name IN ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore']
                    SET d:UrbanDistrict
                    """,
                    
                    """
                    MATCH (d:District)
                    WHERE NOT d.name IN ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore']
                    SET d:RuralDistrict
                    """
                ]
                
                for query in district_relationships:
                    try:
                        session.run(query)
                    except Exception as e:
                        logger.warning(f"District relationship creation failed: {e}")
                
                logger.info("Additional relationships created")
                return True
        
        except Exception as e:
            logger.error(f"Relationship creation failed: {e}")
            return False
    
    def generate_graph_summary(self) -> Dict[str, Any]:
        """Generate summary of graph contents"""
        if not self.driver:
            return {}
        
        try:
            with self.driver.session() as session:
                # Get node counts
                node_counts = {}
                node_types = ['Fact', 'Indicator', 'District', 'Year', 'Source']
                
                for node_type in node_types:
                    result = session.run(f"MATCH (n:{node_type}) RETURN COUNT(n) as count")
                    node_counts[node_type] = result.single()['count']
                
                # Get relationship counts
                rel_counts = {}
                rel_types = ['MEASURED_BY', 'LOCATED_IN', 'OBSERVED_IN', 'REPORTED_BY', 'RELATED_TO']
                
                for rel_type in rel_types:
                    result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) as count")
                    rel_counts[rel_type] = result.single()['count']
                
                # Get top indicators
                result = session.run("""
                    MATCH (f:Fact)-[:MEASURED_BY]->(i:Indicator)
                    RETURN i.name as indicator, COUNT(f) as fact_count
                    ORDER BY fact_count DESC
                    LIMIT 10
                """)
                top_indicators = [dict(record) for record in result]
                
                # Get top districts
                result = session.run("""
                    MATCH (f:Fact)-[:LOCATED_IN]->(d:District)
                    RETURN d.name as district, COUNT(f) as fact_count
                    ORDER BY fact_count DESC
                    LIMIT 10
                """)
                top_districts = [dict(record) for record in result]
                
                return {
                    'node_counts': node_counts,
                    'relationship_counts': rel_counts,
                    'top_indicators': top_indicators,
                    'top_districts': top_districts
                }
        
        except Exception as e:
            logger.error(f"Graph summary generation failed: {e}")
            return {}
    
    def export_graph_data(self) -> bool:
        """Export graph data to files"""
        if not self.driver:
            logger.error("Neo4j driver not available")
            return False
        
        try:
            with self.driver.session() as session:
                # Export nodes
                nodes_query = """
                MATCH (n)
                RETURN labels(n) as labels, properties(n) as properties
                """
                
                result = session.run(nodes_query)
                nodes = [dict(record) for record in result]
                
                nodes_file = self.output_dir / "graph_nodes.json"
                with open(nodes_file, 'w', encoding='utf-8') as f:
                    json.dump(nodes, f, indent=2, ensure_ascii=False)
                
                # Export relationships
                rels_query = """
                MATCH (a)-[r]->(b)
                RETURN type(r) as relationship_type, 
                       labels(a) as start_labels, 
                       properties(a) as start_properties,
                       labels(b) as end_labels, 
                       properties(b) as end_properties
                """
                
                result = session.run(rels_query)
                relationships = [dict(record) for record in result]
                
                rels_file = self.output_dir / "graph_relationships.json"
                with open(rels_file, 'w', encoding='utf-8') as f:
                    json.dump(relationships, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Graph data exported to {self.output_dir}")
                return True
        
        except Exception as e:
            logger.error(f"Graph data export failed: {e}")
            return False
    
    def run_graph_queries(self) -> Dict[str, Any]:
        """Run sample graph queries to demonstrate capabilities"""
        if not self.driver:
            return {}
        
        try:
            with self.driver.session() as session:
                queries = {
                    'ger_by_district': """
                        MATCH (f:Fact)-[:MEASURED_BY]->(i:Indicator {name: 'GER'})
                        MATCH (f)-[:LOCATED_IN]->(d:District)
                        MATCH (f)-[:OBSERVED_IN]->(y:Year)
                        RETURN d.name as district, y.value as year, f.value as ger_value
                        ORDER BY district, year
                    """,
                    
                    'top_performing_districts': """
                        MATCH (f:Fact)-[:MEASURED_BY]->(i:Indicator {name: 'GER'})
                        MATCH (f)-[:LOCATED_IN]->(d:District)
                        MATCH (f)-[:OBSERVED_IN]->(y:Year {value: '2020'})
                        RETURN d.name as district, f.value as ger_value
                        ORDER BY f.value DESC
                        LIMIT 5
                    """,
                    
                    'indicator_trends': """
                        MATCH (f:Fact)-[:MEASURED_BY]->(i:Indicator)
                        MATCH (f)-[:OBSERVED_IN]->(y:Year)
                        RETURN i.name as indicator, y.value as year, AVG(f.value) as avg_value
                        ORDER BY indicator, year
                    """,
                    
                    'source_comparison': """
                        MATCH (f:Fact)-[:REPORTED_BY]->(s:Source)
                        MATCH (f)-[:MEASURED_BY]->(i:Indicator)
                        RETURN s.name as source, i.name as indicator, COUNT(f) as fact_count
                        ORDER BY source, fact_count DESC
                    """
                }
                
                results = {}
                for query_name, query in queries.items():
                    try:
                        result = session.run(query)
                        results[query_name] = [dict(record) for record in result]
                    except Exception as e:
                        logger.warning(f"Query {query_name} failed: {e}")
                        results[query_name] = []
                
                return results
        
        except Exception as e:
            logger.error(f"Graph queries failed: {e}")
            return {}
    
    def close(self):
        """Close Neo4j driver"""
        if self.driver:
            self.driver.close()

def main():
    """Main function to run Neo4j graph loader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load facts to Neo4j')
    parser.add_argument('--normalized-file', default='data/normalized/normalized_facts.json',
                       help='Input file with normalized facts')
    parser.add_argument('--output-dir', default='data/neo4j',
                       help='Output directory for graph data')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load normalized facts
    try:
        with open(args.normalized_file, 'r', encoding='utf-8') as f:
            normalized_facts = json.load(f)
    except FileNotFoundError:
        logger.error(f"Normalized facts file not found: {args.normalized_file}")
        return
    
    # Initialize loader
    loader = Neo4jGraphLoader(output_dir=args.output_dir)
    
    try:
        # Setup graph schema
        if not loader.setup_graph_schema():
            logger.error("Graph schema setup failed")
            return
        
        # Load facts to graph
        if not loader.load_facts_to_graph(normalized_facts):
            logger.error("Fact loading failed")
            return
        
        # Create additional relationships
        if not loader.create_relationships():
            logger.error("Relationship creation failed")
            return
        
        # Generate summary
        summary = loader.generate_graph_summary()
        
        # Export graph data
        loader.export_graph_data()
        
        # Run sample queries
        query_results = loader.run_graph_queries()
        
        # Print summary
        print(f"\nNeo4j Graph Summary:")
        print(f"Nodes: {summary.get('node_counts', {})}")
        print(f"Relationships: {summary.get('relationship_counts', {})}")
        print(f"Top indicators: {len(summary.get('top_indicators', []))}")
        print(f"Top districts: {len(summary.get('top_districts', []))}")
        print(f"Output directory: {args.output_dir}")
        
        # Print sample query results
        if query_results:
            print(f"\nSample Query Results:")
            for query_name, results in query_results.items():
                print(f"{query_name}: {len(results)} results")
    
    finally:
        loader.close()

if __name__ == "__main__":
    main()
