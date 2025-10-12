#!/usr/bin/env python3
"""
Enhanced Stage 3: Build Weaviate Database with Bridge Tables
Loads normalized facts and bridge table connections into Weaviate for hybrid RAG
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import weaviate
from sentence_transformers import SentenceTransformer

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from enhanced_data_normalizer import EnhancedDataNormalizer
from bridge_table_manager import BridgeTableManager

logger = logging.getLogger(__name__)

class WeaviateBridgeLoader:
    """Loads facts and bridge connections into Weaviate for hybrid RAG"""
    
    def __init__(self, weaviate_url: str = "http://localhost:8080"):
        self.weaviate_url = weaviate_url
        self.client = self._init_weaviate()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Schema definitions
        self.fact_schema = {
            "class": "Fact",
            "description": "Education fact with vector embedding",
            "properties": [
                {"name": "fact_id", "dataType": ["text"], "description": "Unique fact identifier"},
                {"name": "district", "dataType": ["text"], "description": "District name"},
                {"name": "indicator", "dataType": ["text"], "description": "Education indicator"},
                {"name": "category", "dataType": ["text"], "description": "Category (SC/ST/OBC/Boys/Girls/etc)"},
                {"name": "year", "dataType": ["int"], "description": "Year of data"},
                {"name": "value", "dataType": ["number"], "description": "Numeric value"},
                {"name": "unit", "dataType": ["text"], "description": "Unit of measurement"},
                {"name": "content", "dataType": ["text"], "description": "Searchable text content"},
                {"name": "source_document", "dataType": ["text"], "description": "Source document"},
                {"name": "source_page", "dataType": ["int"], "description": "Source page number"},
                {"name": "confidence_score", "dataType": ["number"], "description": "Confidence score"},
                {"name": "bridge_connections", "dataType": ["text[]"], "description": "Connected fact IDs from bridge tables"},
                {"name": "metadata", "dataType": ["text"], "description": "Additional metadata as JSON"}
            ],
            "vectorIndexType": "hnsw",
            "vectorizer": "none"  # We'll provide our own vectors
        }
        
        self.bridge_schema = {
            "class": "BridgeTable",
            "description": "Bridge table for fast fact connections",
            "properties": [
                {"name": "bridge_id", "dataType": ["text"], "description": "Bridge table identifier"},
                {"name": "bridge_type", "dataType": ["text"], "description": "Type of bridge"},
                {"name": "key_district", "dataType": ["text"], "description": "District key"},
                {"name": "key_indicator", "dataType": ["text"], "description": "Indicator key"},
                {"name": "key_year", "dataType": ["int"], "description": "Year key"},
                {"name": "connected_facts", "dataType": ["text[]"], "description": "Connected fact IDs"},
                {"name": "summary_stats", "dataType": ["text"], "description": "Summary statistics as JSON"},
                {"name": "indicators_covered", "dataType": ["text[]"], "description": "Indicators in this bridge"},
                {"name": "content", "dataType": ["text"], "description": "Searchable bridge content"},
                {"name": "metadata", "dataType": ["text"], "description": "Bridge metadata as JSON"}
            ],
            "vectorIndexType": "hnsw",
            "vectorizer": "none"
        }
    
    def load_facts_and_bridges(self, facts_file: str = None, bridge_dir: str = None) -> Dict[str, int]:
        """Load facts and bridge tables into Weaviate"""
        logger.info("🚀 Loading facts and bridge tables into Weaviate")
        
        # Setup schema
        self._setup_schema()
        
        # Load facts
        facts_count = self._load_facts(facts_file)
        
        # Load bridge tables
        bridges_count = self._load_bridge_tables(bridge_dir)
        
        # Create cross-references
        self._create_fact_bridge_references()
        
        stats = {
            'facts_loaded': facts_count,
            'bridges_loaded': bridges_count
        }
        
        logger.info(f"✅ Weaviate loading complete: {facts_count} facts, {bridges_count} bridges")
        return stats
    
    def _init_weaviate(self) -> weaviate.Client:
        """Initialize Weaviate client"""
        try:
            client = weaviate.Client(url=self.weaviate_url, timeout_config=(5, 15))
            
            if not client.is_ready():
                raise ConnectionError("Weaviate is not ready")
            
            logger.info(f"✅ Connected to Weaviate at {self.weaviate_url}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    def _setup_schema(self):
        """Setup Weaviate schema"""
        try:
            # Delete existing classes if they exist
            try:
                self.client.schema.delete_class("Fact")
                logger.info("🗑️ Deleted existing Fact class")
            except:
                pass
            
            try:
                self.client.schema.delete_class("BridgeTable")
                logger.info("🗑️ Deleted existing BridgeTable class")
            except:
                pass
            
            # Create new classes
            self.client.schema.create_class(self.fact_schema)
            self.client.schema.create_class(self.bridge_schema)
            
            logger.info("📋 Created Weaviate schema")
            
        except Exception as e:
            logger.error(f"Schema setup failed: {e}")
            raise
    
    def _load_facts(self, facts_file: str = None) -> int:
        """Load normalized facts into Weaviate"""
        if facts_file is None:
            facts_file = "data/normalized/normalized_facts_enhanced.json"
        
        if not Path(facts_file).exists():
            logger.warning(f"Facts file not found: {facts_file}")
            return 0
        
        logger.info(f"📊 Loading facts from {facts_file}")
        
        with open(facts_file, 'r', encoding='utf-8') as f:
            facts = json.load(f)
        
        if not facts:
            logger.warning("No facts to load")
            return 0
        
        # Batch load facts
        with self.client.batch(batch_size=100) as batch:
            for fact in facts:
                try:
                    # Create searchable content
                    content = self._create_fact_content(fact)
                    
                    # Generate embedding
                    embedding = self.embedding_model.encode(content).tolist()
                    
                    # Prepare fact object
                    fact_obj = {
                        "fact_id": fact.get("fact_id", ""),
                        "district": fact.get("district", ""),
                        "indicator": fact.get("indicator", ""),
                        "category": fact.get("category"),
                        "year": fact.get("year", 2023),
                        "value": fact.get("value", 0.0),
                        "unit": fact.get("unit", ""),
                        "content": content,
                        "source_document": fact.get("source_document", ""),
                        "source_page": fact.get("source_page", 1),
                        "confidence_score": fact.get("confidence_score", 0.5),
                        "bridge_connections": [],  # Will be populated later
                        "metadata": json.dumps(fact.get("metadata", {}))
                    }
                    
                    # Add to batch
                    batch.add_data_object(
                        data_object=fact_obj,
                        class_name="Fact",
                        vector=embedding
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to add fact {fact.get('fact_id', 'unknown')}: {e}")
                    continue
        
        logger.info(f"✅ Loaded {len(facts)} facts into Weaviate")
        return len(facts)
    
    def _load_bridge_tables(self, bridge_dir: str = None) -> int:
        """Load bridge tables into Weaviate"""
        if bridge_dir is None:
            bridge_dir = "data/bridge_tables"
        
        bridge_path = Path(bridge_dir)
        if not bridge_path.exists():
            logger.warning(f"Bridge directory not found: {bridge_dir}")
            return 0
        
        logger.info(f"🌉 Loading bridge tables from {bridge_dir}")
        
        total_bridges = 0
        
        # Load different bridge types
        bridge_files = [
            "district_year_bridges.json",
            "indicator_temporal_bridges.json", 
            "policy_hierarchy_bridges.json",
            "cross_district_bridges.json"
        ]
        
        with self.client.batch(batch_size=50) as batch:
            for bridge_file in bridge_files:
                file_path = bridge_path / bridge_file
                if not file_path.exists():
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    bridges = json.load(f)
                
                for bridge_id, bridge_data in bridges.items():
                    try:
                        # Create searchable content for bridge
                        content = self._create_bridge_content(bridge_data)
                        
                        # Generate embedding
                        embedding = self.embedding_model.encode(content).tolist()
                        
                        # Extract key dimensions
                        key_dims = bridge_data.get("key_dimensions", {})
                        
                        # Prepare bridge object
                        bridge_obj = {
                            "bridge_id": bridge_data.get("bridge_id", bridge_id),
                            "bridge_type": bridge_data.get("bridge_type", ""),
                            "key_district": key_dims.get("district"),
                            "key_indicator": key_dims.get("indicator"),
                            "key_year": key_dims.get("year"),
                            "connected_facts": bridge_data.get("connected_facts", []),
                            "summary_stats": json.dumps(bridge_data.get("summary_stats", {})),
                            "indicators_covered": list(bridge_data.get("indicators_covered", [])),
                            "content": content,
                            "metadata": json.dumps(bridge_data.get("metadata", {}))
                        }
                        
                        # Add to batch
                        batch.add_data_object(
                            data_object=bridge_obj,
                            class_name="BridgeTable",
                            vector=embedding
                        )
                        
                        total_bridges += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to add bridge {bridge_id}: {e}")
                        continue
        
        logger.info(f"✅ Loaded {total_bridges} bridge tables into Weaviate")
        return total_bridges
    
    def _create_fact_content(self, fact: Dict[str, Any]) -> str:
        """Create searchable content for fact"""
        parts = []
        
        # Basic fact description
        district = fact.get("district", "")
        indicator = fact.get("indicator", "")
        year = fact.get("year", "")
        value = fact.get("value", "")
        unit = fact.get("unit", "")
        
        if district and indicator:
            parts.append(f"{indicator} in {district}")
        
        if year:
            parts.append(f"Year {year}")
        
        if value is not None and unit:
            parts.append(f"Value: {value} {unit}")
        
        # Add category if available
        category = fact.get("category")
        if category:
            parts.append(f"Category: {category}")
        
        # Add metadata context
        metadata = fact.get("metadata", {})
        if isinstance(metadata, dict):
            if metadata.get("aggregated_from"):
                parts.append(f"Source: {metadata['aggregated_from']}")
        
        return " | ".join(parts)
    
    def _create_bridge_content(self, bridge: Dict[str, Any]) -> str:
        """Create searchable content for bridge table"""
        parts = []
        
        # Bridge type and key dimensions
        bridge_type = bridge.get("bridge_type", "")
        key_dims = bridge.get("key_dimensions", {})
        
        if bridge_type:
            parts.append(f"Bridge type: {bridge_type}")
        
        # Key dimensions
        for key, value in key_dims.items():
            if value:
                parts.append(f"{key}: {value}")
        
        # Summary statistics
        summary = bridge.get("summary_stats", {})
        if isinstance(summary, dict):
            if summary.get("fact_count"):
                parts.append(f"Facts: {summary['fact_count']}")
            if summary.get("district_count"):
                parts.append(f"Districts: {summary['district_count']}")
        
        # Indicators covered
        indicators = bridge.get("indicators_covered", [])
        if indicators:
            parts.append(f"Indicators: {', '.join(indicators[:3])}")
        
        return " | ".join(parts)
    
    def _create_fact_bridge_references(self):
        """Create references between facts and bridge tables"""
        logger.info("🔗 Creating fact-bridge references")
        
        try:
            # Get all bridge tables
            bridges_response = (
                self.client.query
                .get("BridgeTable", ["bridge_id", "connected_facts"])
                .with_limit(1000)
                .do()
            )
            
            bridges = bridges_response.get("data", {}).get("Get", {}).get("BridgeTable", [])
            
            # Update facts with bridge connections
            for bridge in bridges:
                bridge_id = bridge.get("bridge_id", "")
                connected_facts = bridge.get("connected_facts", [])
                
                for fact_id in connected_facts:
                    try:
                        # Get fact by fact_id
                        fact_response = (
                            self.client.query
                            .get("Fact", ["_additional { id }"])
                            .with_where({
                                "path": ["fact_id"],
                                "operator": "Equal",
                                "valueText": fact_id
                            })
                            .with_limit(1)
                            .do()
                        )
                        
                        facts = fact_response.get("data", {}).get("Get", {}).get("Fact", [])
                        if facts:
                            weaviate_id = facts[0]["_additional"]["id"]
                            
                            # Update fact with bridge connection
                            self.client.data_object.update(
                                uuid=weaviate_id,
                                class_name="Fact",
                                data_object={
                                    "bridge_connections": [bridge_id]  # Add bridge_id
                                },
                                merge=True
                            )
                    
                    except Exception as e:
                        logger.warning(f"Failed to update fact {fact_id} with bridge {bridge_id}: {e}")
                        continue
            
            logger.info("✅ Fact-bridge references created")
            
        except Exception as e:
            logger.error(f"Failed to create fact-bridge references: {e}")

def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize loader
        loader = WeaviateBridgeLoader()
        
        # Load facts and bridges
        stats = loader.load_facts_and_bridges()
        
        print(f"🎉 Stage 3 Complete!")
        print(f"   - Facts loaded: {stats['facts_loaded']}")
        print(f"   - Bridges loaded: {stats['bridges_loaded']}")
        
        # Save statistics
        stats_file = Path("data/weaviate/loading_stats.json")
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_file, 'w') as f:
            json.dump({
                **stats,
                'loaded_at': datetime.now().isoformat(),
                'stage': 'weaviate_with_bridges'
            }, f, indent=2)
        
        return 0
        
    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())