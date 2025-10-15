"""
Auto-Registration System for New Datasets
Handles automatic discovery and registration of new data sources
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatasetSchema:
    """Schema definition for a dataset"""
    name: str
    source_type: str  # 'pdf', 'csv', 'json', 'api'
    entity_types: List[str]  # ['school', 'district', 'policy', 'go_number']
    key_fields: Dict[str, str]  # {'school_code': 'string', 'go_number': 'string'}
    bridge_potential: List[str]  # Fields that can link to other datasets
    extraction_method: str  # 'table_extraction', 'nlp_extraction', 'structured'
    confidence_threshold: float = 0.7

class DatasetRegistry:
    """Manages automatic registration and schema detection for new datasets"""
    
    def __init__(self, registry_path: str = "data/registry/"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.schemas_file = self.registry_path / "dataset_schemas.json"
        self.load_schemas()
        
    def load_schemas(self):
        """Load existing dataset schemas"""
        if self.schemas_file.exists():
            with open(self.schemas_file, 'r') as f:
                schema_data = json.load(f)
                self.schemas = {
                    name: DatasetSchema(**schema) 
                    for name, schema in schema_data.items()
                }
        else:
            self.schemas = self._initialize_default_schemas()
            
    def _initialize_default_schemas(self) -> Dict[str, DatasetSchema]:
        """Initialize with known dataset schemas"""
        return {
            'scert': DatasetSchema(
                name='scert',
                source_type='pdf',
                entity_types=['school', 'student', 'application', 'district'],
                key_fields={
                    'school_code': 'string',
                    'district': 'string', 
                    'student_name': 'string'
                },
                bridge_potential=['school_code', 'district'],
                extraction_method='table_extraction'
            ),
            'cse': DatasetSchema(
                name='cse',
                source_type='pdf',
                entity_types=['policy', 'circular', 'district', 'school'],
                key_fields={
                    'policy_number': 'string',
                    'district': 'string',
                    'circular_id': 'string'
                },
                bridge_potential=['district', 'policy_number'],
                extraction_method='nlp_extraction'
            ),
            'udise': DatasetSchema(
                name='udise',
                source_type='csv',
                entity_types=['school', 'enrollment', 'infrastructure', 'district'],
                key_fields={
                    'udise_code': 'string',
                    'district': 'string',
                    'school_name': 'string'
                },
                bridge_potential=['udise_code', 'district'],
                extraction_method='structured'
            ),
            'ap_go': DatasetSchema(
                name='ap_go',
                source_type='pdf',
                entity_types=['go', 'policy', 'department', 'district', 'school'],
                key_fields={
                    'go_number': 'string',
                    'department': 'string',
                    'district': 'string',
                    'date': 'date'
                },
                bridge_potential=['go_number', 'district', 'department'],
                extraction_method='nlp_extraction'
            )
        }
    
    def auto_detect_schema(self, data_path: str, sample_data: Dict[str, Any]) -> DatasetSchema:
        """Automatically detect schema from sample data"""
        logger.info(f"Auto-detecting schema for: {data_path}")
        
        # Extract file info
        path_obj = Path(data_path)
        file_ext = path_obj.suffix.lower()
        
        # Determine source type
        source_type = {
            '.pdf': 'pdf',
            '.csv': 'csv', 
            '.json': 'json',
            '.xlsx': 'excel'
        }.get(file_ext, 'unknown')
        
        # Analyze sample data for entity types
        entity_types = self._detect_entities(sample_data)
        key_fields = self._extract_key_fields(sample_data)
        bridge_potential = self._identify_bridge_fields(key_fields)
        
        # Determine extraction method
        extraction_method = self._determine_extraction_method(source_type, sample_data)
        
        # Generate schema name
        schema_name = self._generate_schema_name(path_obj, entity_types)
        
        return DatasetSchema(
            name=schema_name,
            source_type=source_type,
            entity_types=entity_types,
            key_fields=key_fields,
            bridge_potential=bridge_potential,
            extraction_method=extraction_method
        )
    
    def _detect_entities(self, sample_data: Dict[str, Any]) -> List[str]:
        """Detect entity types from sample data"""
        entities = []
        
        # Check for common patterns
        text_content = str(sample_data).lower()
        
        entity_patterns = {
            'school': ['school', 'institution', 'college'],
            'district': ['district', 'mandal', 'division'],
            'policy': ['policy', 'circular', 'order'],
            'go': ['go.', 'g.o.', 'government order'],
            'student': ['student', 'pupil', 'learner'],
            'teacher': ['teacher', 'faculty', 'staff'],
            'infrastructure': ['building', 'room', 'facility'],
            'enrollment': ['admission', 'enrollment', 'registration']
        }
        
        for entity, patterns in entity_patterns.items():
            if any(pattern in text_content for pattern in patterns):
                entities.append(entity)
        
        return entities or ['document']  # Default if nothing detected
    
    def _extract_key_fields(self, sample_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract key fields and their types"""
        key_fields = {}
        
        if isinstance(sample_data, dict):
            for key, value in sample_data.items():
                field_type = self._infer_field_type(value)
                key_fields[key.lower().replace(' ', '_')] = field_type
        
        return key_fields
    
    def _infer_field_type(self, value: Any) -> str:
        """Infer field type from value"""
        if isinstance(value, int):
            return 'integer'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, str):
            # Check for specific patterns
            if value.isdigit():
                return 'string_numeric'
            elif 'date' in str(value).lower() or len(value) == 10 and '/' in value:
                return 'date'
            else:
                return 'string'
        else:
            return 'unknown'
    
    def _identify_bridge_fields(self, key_fields: Dict[str, str]) -> List[str]:
        """Identify fields that can bridge to other datasets"""
        bridge_patterns = {
            'school_code', 'udise_code', 'school_id',
            'district', 'mandal', 'division',
            'go_number', 'policy_id', 'circular_id',
            'department', 'office', 'institution_id',
            'date', 'year', 'academic_year'
        }
        
        bridge_fields = []
        for field in key_fields.keys():
            if any(pattern in field.lower() for pattern in bridge_patterns):
                bridge_fields.append(field)
        
        return bridge_fields
    
    def _determine_extraction_method(self, source_type: str, sample_data: Dict[str, Any]) -> str:
        """Determine best extraction method"""
        if source_type == 'pdf':
            # Check if it's tabular or text-heavy
            text = str(sample_data).lower()
            if 'table' in text or len(text.split('\n')) > 20:
                return 'table_extraction'
            else:
                return 'nlp_extraction'
        elif source_type in ['csv', 'json', 'excel']:
            return 'structured'
        else:
            return 'nlp_extraction'
    
    def _generate_schema_name(self, path: Path, entities: List[str]) -> str:
        """Generate unique schema name"""
        base_name = path.stem.lower()
        entity_suffix = '_'.join(sorted(entities)[:2])  # Top 2 entities
        return f"{base_name}_{entity_suffix}"
    
    def register_dataset(self, schema: DatasetSchema, overwrite: bool = False) -> bool:
        """Register a new dataset schema"""
        if schema.name in self.schemas and not overwrite:
            logger.warning(f"Schema {schema.name} already exists. Use overwrite=True")
            return False
            
        self.schemas[schema.name] = schema
        self.save_schemas()
        logger.info(f"Registered dataset schema: {schema.name}")
        return True
    
    def save_schemas(self):
        """Save schemas to file"""
        schema_data = {
            name: {
                'name': schema.name,
                'source_type': schema.source_type,
                'entity_types': schema.entity_types,
                'key_fields': schema.key_fields,
                'bridge_potential': schema.bridge_potential,
                'extraction_method': schema.extraction_method,
                'confidence_threshold': schema.confidence_threshold
            }
            for name, schema in self.schemas.items()
        }
        
        with open(self.schemas_file, 'w') as f:
            json.dump(schema_data, f, indent=2)
    
    def get_compatible_schemas(self, target_schema: str) -> List[str]:
        """Find schemas that can bridge to target schema"""
        if target_schema not in self.schemas:
            return []
            
        target = self.schemas[target_schema]
        compatible = []
        
        for name, schema in self.schemas.items():
            if name == target_schema:
                continue
                
            # Check for overlapping bridge fields
            overlap = set(schema.bridge_potential) & set(target.bridge_potential)
            if overlap:
                compatible.append(name)
        
        return compatible
    
    def get_bridge_opportunities(self) -> Dict[str, List[str]]:
        """Get all possible bridge connections"""
        opportunities = {}
        
        for schema_name in self.schemas:
            compatible = self.get_compatible_schemas(schema_name)
            if compatible:
                opportunities[schema_name] = compatible
                
        return opportunities

# Usage example
if __name__ == "__main__":
    registry = DatasetRegistry()
    
    # Example: Register AP GO dataset
    go_schema = DatasetSchema(
        name='ap_go_2024',
        source_type='pdf',
        entity_types=['go', 'policy', 'department'],
        key_fields={'go_number': 'string', 'department': 'string'},
        bridge_potential=['go_number', 'department'],
        extraction_method='nlp_extraction'
    )
    
    registry.register_dataset(go_schema)
    print("Bridge opportunities:", registry.get_bridge_opportunities())