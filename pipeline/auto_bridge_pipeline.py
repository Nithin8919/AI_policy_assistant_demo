"""
Auto-Bridge Pipeline: Automatic Entity Bridging for New Datasets
Handles automatic detection, extraction, and bridging of entities when new data is added
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import os

from pipeline.utils.dataset_registry import DatasetRegistry, DatasetSchema
from pipeline.utils.entity_resolver import EntityResolver
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class AutoBridgePipeline:
    """Automated pipeline for processing new datasets and creating entity bridges"""
    
    def __init__(self, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="password"):
        self.registry = DatasetRegistry()
        self.resolver = EntityResolver()
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Configuration
        self.data_dir = Path("data")
        self.processed_tracker_file = Path("data/registry/processed_datasets.json")
        self.bridge_cache_file = Path("data/registry/bridge_cache.json")
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "registry").mkdir(exist_ok=True)
        
        self.processed_datasets = self.load_processed_tracker()
        self.bridge_cache = self.load_bridge_cache()
        
    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()
    
    def load_processed_tracker(self) -> Dict[str, Any]:
        """Load tracker of processed datasets"""
        if self.processed_tracker_file.exists():
            with open(self.processed_tracker_file, 'r') as f:
                return json.load(f)
        return {"datasets": {}, "last_update": None}
    
    def save_processed_tracker(self):
        """Save tracker of processed datasets"""
        self.processed_datasets["last_update"] = datetime.now().isoformat()
        with open(self.processed_tracker_file, 'w') as f:
            json.dump(self.processed_datasets, f, indent=2)
    
    def load_bridge_cache(self) -> Dict[str, Any]:
        """Load bridge cache for performance"""
        if self.bridge_cache_file.exists():
            with open(self.bridge_cache_file, 'r') as f:
                return json.load(f)
        return {"entity_hashes": {}, "bridge_connections": {}}
    
    def save_bridge_cache(self):
        """Save bridge cache"""
        with open(self.bridge_cache_file, 'w') as f:
            json.dump(self.bridge_cache, f, indent=2)
    
    def detect_new_datasets(self) -> List[Dict[str, Any]]:
        """Detect new datasets in various directories"""
        new_datasets = []
        
        search_paths = [
            "data/raw",
            "data/preprocessed/documents",
            "data/extracted",
            "data/normalized"
        ]
        
        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                continue
                
            for file_path in path.rglob("*"):
                if file_path.is_file() and self._is_processable_file(file_path):
                    file_hash = self._get_file_hash(file_path)
                    file_key = str(file_path.relative_to(Path.cwd()))
                    
                    # Check if this file is new or changed
                    if (file_key not in self.processed_datasets["datasets"] or 
                        self.processed_datasets["datasets"][file_key].get("hash") != file_hash):
                        
                        new_datasets.append({
                            "path": file_path,
                            "key": file_key,
                            "hash": file_hash,
                            "size": file_path.stat().st_size,
                            "modified": file_path.stat().st_mtime
                        })
        
        logger.info(f"Detected {len(new_datasets)} new/changed datasets")
        return new_datasets
    
    def _is_processable_file(self, file_path: Path) -> bool:
        """Check if file is processable"""
        processable_extensions = {'.json', '.csv', '.pdf', '.xlsx', '.txt'}
        return file_path.suffix.lower() in processable_extensions
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file for change detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash {file_path}: {e}")
            return "unknown"
    
    def process_new_dataset(self, dataset_info: Dict[str, Any]) -> Optional[DatasetSchema]:
        """Process a newly detected dataset"""
        file_path = dataset_info["path"]
        logger.info(f"Processing new dataset: {file_path}")
        
        try:
            # Sample the data
            sample_data = self._sample_dataset(file_path)
            
            if not sample_data:
                logger.warning(f"Could not sample data from {file_path}")
                return None
            
            # Auto-detect schema
            schema = self.registry.auto_detect_schema(str(file_path), sample_data)
            
            # Register the dataset
            if self.registry.register_dataset(schema, overwrite=True):
                # Mark as processed
                self.processed_datasets["datasets"][dataset_info["key"]] = {
                    "hash": dataset_info["hash"],
                    "schema_name": schema.name,
                    "processed_at": datetime.now().isoformat(),
                    "status": "processed"
                }
                
                logger.info(f"Registered new dataset schema: {schema.name}")
                return schema
            
        except Exception as e:
            logger.error(f"Failed to process dataset {file_path}: {e}")
            # Mark as failed
            self.processed_datasets["datasets"][dataset_info["key"]] = {
                "hash": dataset_info["hash"],
                "processed_at": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            }
        
        return None
    
    def _sample_dataset(self, file_path: Path) -> Dict[str, Any]:
        """Sample data from various file types"""
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        return data[0]  # First item
                    elif isinstance(data, dict):
                        return data
                    return {"content": str(data)[:500]}
                    
            elif file_path.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path, nrows=1)
                return df.to_dict('records')[0] if not df.empty else {}
                
            elif file_path.suffix.lower() == '.pdf':
                # For PDFs, return metadata
                return {
                    "file_type": "pdf",
                    "name": file_path.stem,
                    "size": file_path.stat().st_size
                }
                
            elif file_path.suffix.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # First 1000 chars
                    return {"content": content}
                    
            else:
                return {"file_type": file_path.suffix, "name": file_path.stem}
                
        except Exception as e:
            logger.warning(f"Could not sample {file_path}: {e}")
            return {"error": str(e), "file_path": str(file_path)}
    
    def extract_entities_from_new_data(self, file_path: Path, schema: DatasetSchema) -> List[Dict[str, Any]]:
        """Extract entities from new dataset based on its schema"""
        logger.info(f"Extracting entities from {file_path} using schema {schema.name}")
        
        try:
            if schema.extraction_method == 'structured':
                return self._extract_structured_entities(file_path, schema)
            elif schema.extraction_method == 'nlp_extraction':
                return self._extract_nlp_entities(file_path, schema)
            elif schema.extraction_method == 'table_extraction':
                return self._extract_table_entities(file_path, schema)
            else:
                logger.warning(f"Unknown extraction method: {schema.extraction_method}")
                return []
                
        except Exception as e:
            logger.error(f"Entity extraction failed for {file_path}: {e}")
            return []
    
    def _extract_structured_entities(self, file_path: Path, schema: DatasetSchema) -> List[Dict[str, Any]]:
        """Extract entities from structured data (JSON, CSV)"""
        entities = []
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        entities = data[:100]  # Limit for demo
                    else:
                        entities = [data]
                        
            elif file_path.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path, nrows=100)  # Limit for demo
                entities = df.to_dict('records')
                
        except Exception as e:
            logger.error(f"Failed to extract structured entities from {file_path}: {e}")
            
        return entities
    
    def _extract_nlp_entities(self, file_path: Path, schema: DatasetSchema) -> List[Dict[str, Any]]:
        """Extract entities using NLP from text-heavy files"""
        # Placeholder for NLP extraction - would use transformers/spaCy
        entities = []
        
        try:
            content = ""
            if file_path.suffix.lower() == '.pdf':
                # Would use PDF extraction library
                content = f"PDF document: {file_path.stem}"
            elif file_path.suffix.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Basic entity extraction (enhance with actual NLP)
            if 'go' in schema.entity_types:
                import re
                go_matches = re.findall(r'G\.O\.?\s*(?:MS\.?\s*)?No\.?\s*(\d+)', content, re.IGNORECASE)
                for go_num in go_matches[:10]:  # Limit
                    entities.append({
                        'go_number': go_num,
                        'source': str(file_path),
                        'type': 'government_order'
                    })
            
            # Extract districts mentioned
            ap_districts = ['Krishna', 'Guntur', 'Prakasam', 'Nellore', 'Chittoor', 
                          'Kadapa', 'Anantapur', 'Kurnool', 'Visakhapatnam', 
                          'Vizianagaram', 'Srikakulam', 'East Godavari', 'West Godavari']
            
            for district in ap_districts:
                if district.lower() in content.lower():
                    entities.append({
                        'district': district,
                        'source': str(file_path),
                        'type': 'district'
                    })
                    
        except Exception as e:
            logger.error(f"NLP extraction failed for {file_path}: {e}")
            
        return entities
    
    def _extract_table_entities(self, file_path: Path, schema: DatasetSchema) -> List[Dict[str, Any]]:
        """Extract entities from tables in PDFs"""
        # Placeholder for table extraction - would use camelot/tabula
        return self._extract_nlp_entities(file_path, schema)  # Fallback to NLP for now
    
    def auto_bridge_new_entities(self, new_entities: List[Dict[str, Any]], schema: DatasetSchema):
        """Automatically create bridges with existing entities"""
        logger.info(f"Creating bridges for {len(new_entities)} new entities from {schema.name}")
        
        # Convert to normalized fact format
        facts = []
        for entity in new_entities:
            fact = {
                'fact_id': f"{schema.name}_{hash(str(entity))}_auto",
                'district': entity.get('district', 'Unknown'),
                'indicator': entity.get('type', 'Unknown'),
                'year': entity.get('year', 2024),
                'value': 1.0,  # Default value
                'source_document': f"{schema.name}_AUTO",
                'metadata': entity
            }
            facts.append(fact)
        
        # Add to resolver
        self.resolver.add_entities_from_facts(facts)
        
        # Resolve new connections
        new_links = self.resolver.resolve_entities()
        
        # Update Neo4j
        self._update_neo4j_with_new_entities(new_links, schema.name)
        
        logger.info(f"Created {len(new_links)} new entity bridges")
        return new_links
    
    def _update_neo4j_with_new_entities(self, new_links, dataset_name):
        """Update Neo4j with new entities and links"""
        with self.neo4j_driver.session() as session:
            # Add new entities
            for entity_key, entity in self.resolver.entities.items():
                if entity.dataset == dataset_name:
                    session.run("""
                        MERGE (e:Entity {id: $id, dataset: $dataset})
                        SET e.type = $type,
                            e.canonical_name = $canonical_name,
                            e.attributes = $attributes,
                            e.created_at = datetime()
                    """, 
                        id=entity.entity_id,
                        type=entity.entity_type,
                        dataset=entity.dataset,
                        canonical_name=entity.canonical_name,
                        attributes=json.dumps(entity.attributes)
                    )
            
            # Add new links
            for link in new_links:
                session.run("""
                    MATCH (e1:Entity {id: $id1, dataset: $dataset1})
                    MATCH (e2:Entity {id: $id2, dataset: $dataset2})
                    MERGE (e1)-[r:ENTITY_LINK {type: $link_type}]->(e2)
                    SET r.confidence = $confidence,
                        r.bridge_field = $bridge_field,
                        r.evidence = $evidence,
                        r.created_at = datetime()
                """,
                    id1=link.entity1_id,
                    dataset1=link.entity1_dataset,
                    id2=link.entity2_id,
                    dataset2=link.entity2_dataset,
                    link_type=link.link_type,
                    confidence=link.confidence_score,
                    bridge_field=link.bridge_field,
                    evidence=json.dumps(link.evidence)
                )
    
    def run_auto_bridge_cycle(self):
        """Run complete auto-bridging cycle"""
        logger.info("Starting auto-bridge cycle...")
        
        try:
            # Step 1: Detect new datasets
            new_datasets = self.detect_new_datasets()
            
            if not new_datasets:
                logger.info("No new datasets detected")
                return {"status": "success", "message": "No new datasets"}
            
            results = {
                "processed": 0,
                "failed": 0,
                "new_entities": 0,
                "new_bridges": 0,
                "datasets": []
            }
            
            # Step 2: Process each new dataset
            for dataset_info in new_datasets:
                try:
                    # Process dataset
                    schema = self.process_new_dataset(dataset_info)
                    
                    if schema:
                        # Extract entities
                        entities = self.extract_entities_from_new_data(dataset_info["path"], schema)
                        
                        if entities:
                            # Create bridges
                            new_links = self.auto_bridge_new_entities(entities, schema)
                            
                            results["processed"] += 1
                            results["new_entities"] += len(entities)
                            results["new_bridges"] += len(new_links)
                            
                            results["datasets"].append({
                                "name": schema.name,
                                "path": str(dataset_info["path"]),
                                "entities": len(entities),
                                "bridges": len(new_links),
                                "status": "success"
                            })
                            
                        else:
                            logger.warning(f"No entities extracted from {dataset_info['path']}")
                            results["failed"] += 1
                    else:
                        results["failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process {dataset_info['path']}: {e}")
                    results["failed"] += 1
                    
                    results["datasets"].append({
                        "path": str(dataset_info["path"]),
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Step 3: Save state
            self.save_processed_tracker()
            self.save_bridge_cache()
            
            logger.info(f"Auto-bridge cycle complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Auto-bridge cycle failed: {e}")
            return {"status": "error", "message": str(e)}

# CLI Interface
def main():
    """Main CLI for auto-bridge pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-Bridge Pipeline for New Datasets")
    parser.add_argument("--run", action="store_true", help="Run auto-bridge cycle")
    parser.add_argument("--detect", action="store_true", help="Detect new datasets only")
    parser.add_argument("--status", action="store_true", help="Show status")
    
    args = parser.parse_args()
    
    pipeline = AutoBridgePipeline()
    
    try:
        if args.run:
            results = pipeline.run_auto_bridge_cycle()
            print(f"\n🚀 Auto-Bridge Results:")
            print(f"   Processed: {results.get('processed', 0)}")
            print(f"   Failed: {results.get('failed', 0)}")
            print(f"   New Entities: {results.get('new_entities', 0)}")
            print(f"   New Bridges: {results.get('new_bridges', 0)}")
            
        elif args.detect:
            new_datasets = pipeline.detect_new_datasets()
            print(f"\n📂 Detected {len(new_datasets)} new/changed datasets:")
            for ds in new_datasets:
                print(f"   {ds['key']} (size: {ds['size']} bytes)")
                
        elif args.status:
            processed = pipeline.processed_datasets
            print(f"\n📊 Status:")
            print(f"   Total Processed: {len(processed.get('datasets', {}))}")
            print(f"   Last Update: {processed.get('last_update', 'Never')}")
            
        else:
            parser.print_help()
            
    finally:
        pipeline.close()

if __name__ == "__main__":
    main()