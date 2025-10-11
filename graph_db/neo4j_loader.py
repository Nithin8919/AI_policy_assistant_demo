"""
Neo4j Knowledge Graph Loader and Manager
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Graph node structure"""
    node_id: str
    node_type: str
    properties: Dict[str, Any]
    labels: List[str]

@dataclass
class GraphRelation:
    """Graph relation structure"""
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]

class Neo4jGraphLoader:
    """Neo4j knowledge graph loader and manager"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple = ("neo4j", "password")):
        self.uri = uri
        self.auth = auth
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self._test_connection()
        self._create_constraints()
    
    def _test_connection(self):
        """Test Neo4j connection"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection successful")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise
    
    def _create_constraints(self):
        """Create unique constraints for graph nodes"""
        constraints = [
            "CREATE CONSTRAINT FOR (a:Act) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (r:Rule) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (j:Judgment) REQUIRE j.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (p:Policy) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (o:Organization) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (per:Person) REQUIRE per.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (loc:Location) REQUIRE loc.id IS UNIQUE"
        ]
        
        try:
            with self.driver.session() as session:
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        # Constraint might already exist
                        logger.debug(f"Constraint creation skipped: {e}")
            
            logger.info("Graph constraints created/verified")
        except Exception as e:
            logger.error(f"Failed to create constraints: {e}")
    
    def load_ontology_schema(self, schema_file: str = "graph_db/ontology_schema.cql"):
        """Load ontology schema from CQL file"""
        try:
            schema_path = Path(schema_file)
            if not schema_path.exists():
                logger.warning(f"Schema file not found: {schema_file}")
                return
            
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_queries = f.read().split(';')
            
            with self.driver.session() as session:
                for query in schema_queries:
                    query = query.strip()
                    if query:
                        try:
                            session.run(query)
                        except Exception as e:
                            logger.debug(f"Schema query skipped: {e}")
            
            logger.info("Ontology schema loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ontology schema: {e}")
    
    def create_node(self, node: GraphNode) -> bool:
        """
        Create a node in the graph
        
        Args:
            node: GraphNode object
            
        Returns:
            True if successful
        """
        try:
            with self.driver.session() as session:
                # Create labels string
                labels_str = ":".join(node.labels) if node.labels else "Entity"
                
                # Create properties
                props = node.properties.copy()
                props['id'] = node.node_id
                
                query = f"""
                MERGE (n:{labels_str} {{id: $id}})
                SET n += $properties
                RETURN n
                """
                
                result = session.run(query, id=node.node_id, properties=props)
                result.single()
                
            logger.debug(f"Created node: {node.node_id} ({labels_str})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create node {node.node_id}: {e}")
            return False
    
    def create_relation(self, relation: GraphRelation) -> bool:
        """
        Create a relation in the graph
        
        Args:
            relation: GraphRelation object
            
        Returns:
            True if successful
        """
        try:
            with self.driver.session() as session:
                props = relation.properties.copy()
                props['id'] = relation.relation_id
                props['type'] = relation.relation_type
                
                query = """
                MATCH (a {id: $source_id})
                MATCH (b {id: $target_id})
                MERGE (a)-[r:RELATION {id: $relation_id}]->(b)
                SET r += $properties
                RETURN r
                """
                
                result = session.run(
                    query,
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    relation_id=relation.relation_id,
                    properties=props
                )
                result.single()
                
            logger.debug(f"Created relation: {relation.source_id} -> {relation.target_id} ({relation.relation_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create relation {relation.relation_id}: {e}")
            return False
    
    def load_entities_from_nlp(self, nlp_results: Dict[str, Any]) -> int:
        """
        Load entities from NLP processing results
        
        Args:
            nlp_results: NLP processing results
            
        Returns:
            Number of entities loaded
        """
        loaded_count = 0
        
        try:
            doc_id = nlp_results.get('doc_id')
            entities = nlp_results.get('entities', [])
            
            # Create document node
            doc_node = GraphNode(
                node_id=doc_id,
                node_type="Document",
                properties={
                    'name': doc_id,
                    'type': 'policy_document',
                    'source': 'nlp_processing'
                },
                labels=['Document']
            )
            
            if self.create_node(doc_node):
                loaded_count += 1
            
            # Create entity nodes
            for entity_data in entities:
                entity_id = entity_data['entity_id']
                entity_text = entity_data['text']
                entity_label = entity_data['label']
                
                # Map NLP labels to graph node types
                node_type = self._map_entity_label_to_node_type(entity_label)
                
                entity_node = GraphNode(
                    node_id=entity_id,
                    node_type=node_type,
                    properties={
                        'name': entity_text,
                        'text': entity_text,
                        'label': entity_label,
                        'confidence': entity_data['confidence'],
                        'start_pos': entity_data['start'],
                        'end_pos': entity_data['end'],
                        'context': entity_data.get('context', '')
                    },
                    labels=[node_type, 'Entity']
                )
                
                if self.create_node(entity_node):
                    loaded_count += 1
                
                # Create relation between document and entity
                doc_entity_relation = GraphRelation(
                    relation_id=f"DOC_ENTITY_{doc_id}_{entity_id}",
                    source_id=doc_id,
                    target_id=entity_id,
                    relation_type="CONTAINS",
                    properties={
                        'confidence': entity_data['confidence'],
                        'position': entity_data['start']
                    }
                )
                
                self.create_relation(doc_entity_relation)
            
            logger.info(f"Loaded {loaded_count} entities from NLP results for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to load entities from NLP results: {e}")
        
        return loaded_count
    
    def load_relations_from_nlp(self, nlp_results: Dict[str, Any]) -> int:
        """
        Load relations from NLP processing results
        
        Args:
            nlp_results: NLP processing results
            
        Returns:
            Number of relations loaded
        """
        loaded_count = 0
        
        try:
            relations = nlp_results.get('relations', [])
            
            for relation_data in relations:
                relation_id = relation_data['relation_id']
                head_entity_id = relation_data['head_entity_id']
                tail_entity_id = relation_data['tail_entity_id']
                relation_type = relation_data['relation_type']
                
                relation = GraphRelation(
                    relation_id=relation_id,
                    source_id=head_entity_id,
                    target_id=tail_entity_id,
                    relation_type=relation_type,
                    properties={
                        'confidence': relation_data['confidence'],
                        'context': relation_data.get('context', '')
                    }
                )
                
                if self.create_relation(relation):
                    loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} relations from NLP results")
            
        except Exception as e:
            logger.error(f"Failed to load relations from NLP results: {e}")
        
        return loaded_count
    
    def _map_entity_label_to_node_type(self, label: str) -> str:
        """Map NLP entity label to graph node type"""
        mapping = {
            'PERSON': 'Person',
            'ORG': 'Organization',
            'GPE': 'Location',
            'LAW': 'Act',
            'GO_NUMBER': 'Policy',
            'CIRCULAR_NUMBER': 'Policy',
            'DATE': 'Date',
            'AMOUNT': 'Amount',
            'PERCENTAGE': 'Percentage',
            'SCHOOL_TYPE': 'SchoolType'
        }
        
        return mapping.get(label, 'Entity')
    
    def bulk_load_from_json(self, json_file: str) -> Dict[str, int]:
        """
        Bulk load nodes and relations from JSON file
        
        Args:
            json_file: Path to JSON file containing graph data
            
        Returns:
            Dictionary with loading statistics
        """
        stats = {'nodes_loaded': 0, 'relations_loaded': 0, 'errors': 0}
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load nodes
            nodes = data.get('nodes', [])
            for node_data in nodes:
                node = GraphNode(
                    node_id=node_data['id'],
                    node_type=node_data.get('type', 'Entity'),
                    properties=node_data.get('properties', {}),
                    labels=node_data.get('labels', ['Entity'])
                )
                
                if self.create_node(node):
                    stats['nodes_loaded'] += 1
                else:
                    stats['errors'] += 1
            
            # Load relations
            relations = data.get('relations', [])
            for relation_data in relations:
                relation = GraphRelation(
                    relation_id=relation_data['id'],
                    source_id=relation_data['source'],
                    target_id=relation_data['target'],
                    relation_type=relation_data['type'],
                    properties=relation_data.get('properties', {})
                )
                
                if self.create_relation(relation):
                    stats['relations_loaded'] += 1
                else:
                    stats['errors'] += 1
            
            logger.info(f"Bulk loaded {stats['nodes_loaded']} nodes and {stats['relations_loaded']} relations")
            
        except Exception as e:
            logger.error(f"Failed to bulk load from JSON: {e}")
            stats['errors'] += 1
        
        return stats
    
    def query_graph(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query on the graph
        
        Args:
            cypher_query: Cypher query string
            
        Returns:
            Query results
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = []
                
                for record in result:
                    records.append(dict(record))
                
                logger.info(f"Executed query, returned {len(records)} records")
                return records
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def get_entity_context(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get context around an entity
        
        Args:
            entity_id: Entity identifier
            depth: Traversal depth
            
        Returns:
            Entity context information
        """
        try:
            query = f"""
            MATCH (e {{id: $entity_id}})
            OPTIONAL MATCH path = (e)-[r*1..{depth}]-(connected)
            RETURN e, path, relationships(path) as rels
            """
            
            results = self.query_graph(query, entity_id=entity_id)
            
            if results:
                result = results[0]
                entity_data = dict(result['e']) if result['e'] else {}
                
                # Extract connected nodes and relationships
                connected_nodes = []
                relationships = []
                
                if result['path']:
                    for node in result['path'].nodes:
                        connected_nodes.append({
                            'id': node.get('id'),
                            'labels': list(node.labels),
                            'properties': dict(node)
                        })
                    
                    for rel in result['rels']:
                        relationships.append({
                            'type': rel.type,
                            'properties': dict(rel)
                        })
                
                return {
                    'entity': entity_data,
                    'connected_nodes': connected_nodes,
                    'relationships': relationships
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get entity context: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        try:
            stats = {}
            
            # Count nodes by label
            node_query = """
            CALL db.labels() YIELD label
            CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value
            RETURN label, value.count as count
            """
            
            try:
                results = self.query_graph(node_query)
                node_counts = {record['label']: record['count'] for record in results}
                stats['node_counts'] = node_counts
            except:
                # Fallback if APOC is not available
                labels = ['Act', 'Rule', 'Policy', 'Entity', 'Document', 'Judgment', 'Organization', 'Person', 'Location']
                node_counts = {}
                for label in labels:
                    try:
                        result = self.query_graph(f"MATCH (n:{label}) RETURN count(n) as count")
                        if result:
                            node_counts[label] = result[0]['count']
                    except:
                        node_counts[label] = 0
                stats['node_counts'] = node_counts
            
            # Count relationships
            rel_query = "MATCH ()-[r]->() RETURN count(r) as total_relations"
            result = self.query_graph(rel_query)
            stats['total_relations'] = result[0]['total_relations'] if result else 0
            
            # Count total nodes
            node_query = "MATCH (n) RETURN count(n) as total_nodes"
            result = self.query_graph(node_query)
            stats['total_nodes'] = result[0]['total_nodes'] if result else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {}
    
    def export_graph(self, output_file: str):
        """Export graph data to JSON file"""
        try:
            # Export nodes
            nodes_query = """
            MATCH (n)
            RETURN n.id as id, labels(n) as labels, properties(n) as properties
            """
            nodes = self.query_graph(nodes_query)
            
            # Export relations
            relations_query = """
            MATCH (a)-[r]->(b)
            RETURN r.id as id, a.id as source, b.id as target, r.type as type, properties(r) as properties
            """
            relations = self.query_graph(relations_query)
            
            # Prepare export data
            export_data = {
                'nodes': nodes,
                'relations': relations,
                'statistics': self.get_statistics()
            }
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported graph data to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

def main():
    """Test the Neo4j graph loader"""
    loader = Neo4jGraphLoader()
    
    try:
        # Test connection
        print("Testing Neo4j connection...")
        stats = loader.get_statistics()
        print(f"Graph statistics: {stats}")
        
        # Test node creation
        test_node = GraphNode(
            node_id="test_node_1",
            node_type="Test",
            properties={'name': 'Test Node', 'value': 123},
            labels=['Test', 'Entity']
        )
        
        if loader.create_node(test_node):
            print("Test node created successfully")
        
        # Test relation creation
        test_relation = GraphRelation(
            relation_id="test_rel_1",
            source_id="test_node_1",
            target_id="test_node_1",
            relation_type="SELF_RELATED",
            properties={'confidence': 1.0}
        )
        
        if loader.create_relation(test_relation):
            print("Test relation created successfully")
        
        # Get statistics
        final_stats = loader.get_statistics()
        print(f"Final statistics: {final_stats}")
        
    finally:
        loader.close()

if __name__ == "__main__":
    main()
