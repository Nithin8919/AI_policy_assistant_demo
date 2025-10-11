"""
Graph Manager for Neo4j Knowledge Graph
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from pathlib import Path
import os

# Optional import
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)

class GraphManager:
    """Manages Neo4j knowledge graph operations"""
    
    def __init__(self, uri: Optional[str] = None, auth: Optional[Tuple[str, str]] = None):
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available. Graph operations will be disabled.")
            self.driver = None
            return
            
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.auth = auth or (
            os.getenv('NEO4J_USER', 'neo4j'),
            os.getenv('NEO4J_PASSWORD', 'password')
        )
        
        self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
        self._test_connection()
        self._create_constraints()
    
    def _test_connection(self):
        """Test Neo4j connection"""
        if not NEO4J_AVAILABLE or self.driver is None:
            logger.warning("Neo4j not available, skipping connection test")
            return
            
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection successful")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise
    
    def _create_constraints(self):
        """Create unique constraints for graph nodes"""
        if not NEO4J_AVAILABLE or self.driver is None:
            logger.warning("Neo4j not available, skipping constraint creation")
            return
            
        constraints = [
            "CREATE CONSTRAINT FOR (a:Act) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (r:Rule) REQUIRE r.id IS UNIQUE", 
            "CREATE CONSTRAINT FOR (j:Judgment) REQUIRE j.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (p:Policy) REQUIRE p.id IS UNIQUE"
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
    
    def create_entity(
        self, 
        entity_id: str, 
        entity_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create or update an entity node
        
        Args:
            entity_id: Unique entity identifier
            entity_type: Type of entity (Act, Rule, Policy, etc.)
            properties: Additional properties for the entity
            
        Returns:
            True if successful
        """
        if not NEO4J_AVAILABLE or self.driver is None:
            logger.warning("Neo4j not available, skipping entity creation")
            return False
            
        try:
            with self.driver.session() as session:
                props = properties or {}
                props['id'] = entity_id
                props['type'] = entity_type
                
                query = f"""
                MERGE (e:{entity_type} {{id: $id}})
                SET e += $properties
                RETURN e
                """
                
                result = session.run(query, id=entity_id, properties=props)
                result.single()
                
            logger.info(f"Created/updated entity: {entity_id} ({entity_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create entity {entity_id}: {e}")
            return False
    
    def create_relation(
        self,
        head_entity_id: str,
        tail_entity_id: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a relationship between two entities
        
        Args:
            head_entity_id: Source entity ID
            tail_entity_id: Target entity ID
            relation_type: Type of relationship
            properties: Additional properties for the relationship
            
        Returns:
            True if successful
        """
        try:
            with self.driver.session() as session:
                props = properties or {}
                
                query = """
                MATCH (a {id: $head_id})
                MATCH (b {id: $tail_id})
                MERGE (a)-[r:RELATION {type: $rel_type}]->(b)
                SET r += $properties
                RETURN r
                """
                
                result = session.run(
                    query,
                    head_id=head_entity_id,
                    tail_id=tail_entity_id,
                    rel_type=relation_type,
                    properties=props
                )
                result.single()
                
            logger.info(f"Created relation: {head_entity_id} -> {tail_entity_id} ({relation_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create relation: {e}")
            return False
    
    def get_entity_context(
        self, 
        entity_ids: List[str], 
        depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get context around entities (neighboring nodes and relationships)
        
        Args:
            entity_ids: List of entity IDs to get context for
            depth: How many hops to traverse
            
        Returns:
            List of context information
        """
        try:
            with self.driver.session() as session:
                context = []
                
                for entity_id in entity_ids:
                    query = f"""
                    MATCH (e {{id: $entity_id}})
                    OPTIONAL MATCH path = (e)-[r*1..{depth}]-(connected)
                    RETURN e, path, relationships(path) as rels
                    """
                    
                    result = session.run(query, entity_id=entity_id)
                    
                    for record in result:
                        entity_data = dict(record['e'])
                        path_data = []
                        
                        if record['path']:
                            for node in record['path'].nodes:
                                path_data.append({
                                    'id': node.get('id'),
                                    'type': list(node.labels)[0] if node.labels else 'Unknown',
                                    'properties': dict(node)
                                })
                        
                        context.append({
                            'entity_id': entity_id,
                            'entity_data': entity_data,
                            'connected_nodes': path_data,
                            'relationships': [dict(rel) for rel in record['rels']] if record['rels'] else []
                        })
                
                logger.info(f"Retrieved context for {len(entity_ids)} entities")
                return context
                
        except Exception as e:
            logger.error(f"Failed to get entity context: {e}")
            return []
    
    def find_related_entities(
        self, 
        entity_id: str, 
        relation_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity
        
        Args:
            entity_id: Entity to find relations for
            relation_types: Specific relation types to filter by
            max_results: Maximum number of results
            
        Returns:
            List of related entities
        """
        try:
            with self.driver.session() as session:
                if relation_types:
                    rel_filter = f"WHERE type(r) IN {relation_types}"
                else:
                    rel_filter = ""
                
                query = f"""
                MATCH (e {{id: $entity_id}})-[r]-(related)
                {rel_filter}
                RETURN related, type(r) as relation_type, r
                LIMIT $max_results
                """
                
                result = session.run(query, entity_id=entity_id, max_results=max_results)
                
                related_entities = []
                for record in result:
                    related_entities.append({
                        'entity_id': record['related'].get('id'),
                        'entity_type': list(record['related'].labels)[0] if record['related'].labels else 'Unknown',
                        'relation_type': record['relation_type'],
                        'properties': dict(record['related']),
                        'relation_properties': dict(record['r'])
                    })
                
                logger.info(f"Found {len(related_entities)} related entities for {entity_id}")
                return related_entities
                
        except Exception as e:
            logger.error(f"Failed to find related entities: {e}")
            return []
    
    def get_entity_paths(
        self, 
        start_entity_id: str, 
        end_entity_id: str, 
        max_length: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find paths between two entities
        
        Args:
            start_entity_id: Starting entity ID
            end_entity_id: Ending entity ID
            max_length: Maximum path length
            
        Returns:
            List of paths between entities
        """
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (start {{id: $start_id}}), (end {{id: $end_id}})
                MATCH path = shortestPath((start)-[*1..{max_length}]-(end))
                RETURN path, length(path) as path_length
                """
                
                result = session.run(query, start_id=start_entity_id, end_id=end_entity_id)
                
                paths = []
                for record in result:
                    path_nodes = []
                    path_rels = []
                    
                    for node in record['path'].nodes:
                        path_nodes.append({
                            'id': node.get('id'),
                            'type': list(node.labels)[0] if node.labels else 'Unknown',
                            'properties': dict(node)
                        })
                    
                    for rel in record['path'].relationships:
                        path_rels.append({
                            'type': rel.type,
                            'properties': dict(rel)
                        })
                    
                    paths.append({
                        'path_length': record['path_length'],
                        'nodes': path_nodes,
                        'relationships': path_rels
                    })
                
                logger.info(f"Found {len(paths)} paths between {start_entity_id} and {end_entity_id}")
                return paths
                
        except Exception as e:
            logger.error(f"Failed to find entity paths: {e}")
            return []
    
    def bulk_create_entities(
        self, 
        entities: List[Dict[str, Any]]
    ) -> int:
        """
        Bulk create entities from a list
        
        Args:
            entities: List of entity dictionaries with id, type, and properties
            
        Returns:
            Number of entities created
        """
        try:
            with self.driver.session() as session:
                created_count = 0
                
                for entity in entities:
                    entity_id = entity.get('id')
                    entity_type = entity.get('type', 'Entity')
                    properties = entity.get('properties', {})
                    
                    if entity_id:
                        if self.create_entity(entity_id, entity_type, properties):
                            created_count += 1
                
                logger.info(f"Bulk created {created_count} entities")
                return created_count
                
        except Exception as e:
            logger.error(f"Bulk entity creation failed: {e}")
            return 0
    
    def bulk_create_relations(
        self, 
        relations: List[Dict[str, Any]]
    ) -> int:
        """
        Bulk create relationships from a list
        
        Args:
            relations: List of relation dictionaries
            
        Returns:
            Number of relations created
        """
        try:
            with self.driver.session() as session:
                created_count = 0
                
                for relation in relations:
                    head_id = relation.get('head_entity_id')
                    tail_id = relation.get('tail_entity_id')
                    rel_type = relation.get('relation_type')
                    properties = relation.get('properties', {})
                    
                    if all([head_id, tail_id, rel_type]):
                        if self.create_relation(head_id, tail_id, rel_type, properties):
                            created_count += 1
                
                logger.info(f"Bulk created {created_count} relations")
                return created_count
                
        except Exception as e:
            logger.error(f"Bulk relation creation failed: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        try:
            with self.driver.session() as session:
                stats = {}
                
                # Count nodes by label
                node_query = """
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value
                RETURN label, value.count as count
                """
                
                try:
                    result = session.run(node_query)
                    node_counts = {record['label']: record['count'] for record in result}
                    stats['node_counts'] = node_counts
                except:
                    # Fallback if APOC is not available
                    labels = ['Act', 'Rule', 'Policy', 'Entity', 'Document', 'Judgment']
                    node_counts = {}
                    for label in labels:
                        try:
                            result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                            count = result.single()['count']
                            node_counts[label] = count
                        except:
                            node_counts[label] = 0
                    stats['node_counts'] = node_counts
                
                # Count relationships
                rel_query = "MATCH ()-[r]->() RETURN count(r) as total_relations"
                result = session.run(rel_query)
                stats['total_relations'] = result.single()['total_relations']
                
                # Count total nodes
                node_query = "MATCH (n) RETURN count(n) as total_nodes"
                result = session.run(node_query)
                stats['total_nodes'] = result.single()['total_nodes']
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {}
    
    def check_connection(self) -> bool:
        """Check if graph database connection is healthy"""
        if not NEO4J_AVAILABLE or self.driver is None:
            return False
            
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except:
            return False
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

