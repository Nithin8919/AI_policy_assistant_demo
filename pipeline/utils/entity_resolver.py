"""
LLM-Powered Entity Resolution System
Creates meaningful connections between entities across different datasets
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re
from difflib import SequenceMatcher
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class EntityLink:
    """Represents a link between two entities"""
    entity1_id: str
    entity2_id: str
    entity1_dataset: str
    entity2_dataset: str
    link_type: str  # 'same_entity', 'related', 'hierarchical', 'temporal'
    confidence_score: float
    bridge_field: str
    evidence: Dict[str, Any]

@dataclass
class Entity:
    """Represents an entity across datasets"""
    entity_id: str
    dataset: str
    entity_type: str  # 'school', 'district', 'policy', etc.
    attributes: Dict[str, Any]
    canonical_name: str
    aliases: List[str]

class EntityResolver:
    """Resolves entities across different datasets using multiple strategies"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.entity_links: List[EntityLink] = []
        self.resolution_strategies = [
            self._exact_match_strategy,
            self._fuzzy_match_strategy,
            self._code_match_strategy,
            self._geographic_strategy,
            self._temporal_strategy,
            self._semantic_strategy
        ]
    
    def add_entities_from_facts(self, facts: List[Dict[str, Any]]) -> None:
        """Extract entities from normalized facts"""
        logger.info(f"Processing {len(facts)} facts for entity extraction")
        
        for fact in facts:
            # Extract different entity types from each fact
            entities = self._extract_entities_from_fact(fact)
            
            for entity in entities:
                entity_key = f"{entity.dataset}_{entity.entity_type}_{entity.entity_id}"
                
                if entity_key in self.entities:
                    # Merge with existing entity
                    self._merge_entities(self.entities[entity_key], entity)
                else:
                    self.entities[entity_key] = entity
    
    def _extract_entities_from_fact(self, fact: Dict[str, Any]) -> List[Entity]:
        """Extract all entities from a single fact"""
        entities = []
        dataset = fact.get('source_document', '').split('_')[0].lower()
        
        # District entity
        if fact.get('district') and fact['district'] != 'Unknown':
            entities.append(Entity(
                entity_id=f"district_{fact['district']}_{dataset}",
                dataset=dataset,
                entity_type='district',
                attributes={
                    'name': fact['district'],
                    'state': 'Andhra Pradesh'
                },
                canonical_name=fact['district'].title(),
                aliases=[fact['district'].upper(), fact['district'].lower()]
            ))
        
        # School entities from metadata
        if 'metadata' in fact and fact['metadata']:
            metadata = fact['metadata']
            
            # Extract school codes and names
            school_codes = metadata.get('school_codes', [])
            school_names = self._extract_school_names(metadata.get('sample_data', []))
            
            for i, school_code in enumerate(school_codes[:10]):  # Limit for demo
                school_name = school_names[i] if i < len(school_names) else f"School_{school_code}"
                
                entities.append(Entity(
                    entity_id=f"school_{school_code}_{dataset}",
                    dataset=dataset,
                    entity_type='school',
                    attributes={
                        'code': school_code,
                        'name': school_name,
                        'district': fact.get('district'),
                        'year': fact.get('year')
                    },
                    canonical_name=school_name,
                    aliases=[school_code, school_name.upper()]
                ))
        
        # Policy/GO entities
        if 'go_number' in str(fact.get('source_document', '')):
            go_match = re.search(r'GO[.\s]*(\d+)', fact.get('source_document', ''), re.IGNORECASE)
            if go_match:
                go_number = go_match.group(1)
                entities.append(Entity(
                    entity_id=f"go_{go_number}",
                    dataset=dataset,
                    entity_type='policy',
                    attributes={
                        'go_number': go_number,
                        'year': fact.get('year'),
                        'district': fact.get('district')
                    },
                    canonical_name=f"G.O. {go_number}",
                    aliases=[f"GO {go_number}", f"G.O.{go_number}"]
                ))
        
        return entities
    
    def _extract_school_names(self, sample_data: List[str]) -> List[str]:
        """Extract school names from sample data strings"""
        school_names = []
        
        for data_str in sample_data:
            # Common patterns for school names
            patterns = [
                r'([A-Z][A-Z\s]+(?:SCHOOL|COLLEGE|ACADEMY))',
                r'(\w+\s+(?:EM|ENGLISH|HIGH)\s+SCHOOL)',
                r'(SRI\s+\w+\s+\w+\s+SCHOOL)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, data_str, re.IGNORECASE)
                school_names.extend(matches)
                if matches:
                    break  # Found a match, move to next data string
        
        return [name.title() for name in school_names]
    
    def resolve_entities(self) -> List[EntityLink]:
        """Apply all resolution strategies to find entity links"""
        logger.info("Starting entity resolution process")
        
        entity_list = list(self.entities.values())
        total_comparisons = len(entity_list) * (len(entity_list) - 1) // 2
        logger.info(f"Processing {len(entity_list)} entities, {total_comparisons} comparisons")
        
        links = []
        
        # Compare each pair of entities
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                if entity1.dataset == entity2.dataset:
                    continue  # Skip same-dataset comparisons
                
                # Apply all strategies
                for strategy in self.resolution_strategies:
                    link = strategy(entity1, entity2)
                    if link and link.confidence_score > 0.6:  # Threshold
                        links.append(link)
                        break  # Use first successful strategy
        
        self.entity_links = links
        logger.info(f"Found {len(links)} entity links")
        return links
    
    def _exact_match_strategy(self, entity1: Entity, entity2: Entity) -> Optional[EntityLink]:
        """Find exact matches between entity identifiers"""
        evidence = {}
        confidence = 0.0
        link_type = 'same_entity'
        bridge_field = 'exact_match'
        
        # Check exact code matches
        code1 = entity1.attributes.get('code', entity1.entity_id)
        code2 = entity2.attributes.get('code', entity2.entity_id)
        
        if code1 == code2 and len(code1) > 5:  # Meaningful code match
            confidence = 0.95
            evidence['matching_code'] = code1
            bridge_field = 'code'
        
        # Check exact name matches
        elif entity1.canonical_name.lower() == entity2.canonical_name.lower():
            confidence = 0.85
            evidence['matching_name'] = entity1.canonical_name
            bridge_field = 'name'
        
        if confidence > 0.6:
            return EntityLink(
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                entity1_dataset=entity1.dataset,
                entity2_dataset=entity2.dataset,
                link_type=link_type,
                confidence_score=confidence,
                bridge_field=bridge_field,
                evidence=evidence
            )
        
        return None
    
    def _fuzzy_match_strategy(self, entity1: Entity, entity2: Entity) -> Optional[EntityLink]:
        """Find fuzzy matches between entity names"""
        if entity1.entity_type != entity2.entity_type:
            return None
            
        similarity = SequenceMatcher(None, 
            entity1.canonical_name.lower(), 
            entity2.canonical_name.lower()
        ).ratio()
        
        if similarity > 0.8:
            return EntityLink(
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                entity1_dataset=entity1.dataset,
                entity2_dataset=entity2.dataset,
                link_type='same_entity',
                confidence_score=similarity * 0.9,  # Slightly lower than exact
                bridge_field='name_similarity',
                evidence={'similarity_score': similarity, 'names': [entity1.canonical_name, entity2.canonical_name]}
            )
        
        return None
    
    def _code_match_strategy(self, entity1: Entity, entity2: Entity) -> Optional[EntityLink]:
        """Match entities based on various code patterns"""
        codes1 = [entity1.attributes.get('code'), entity1.attributes.get('udise_code'), 
                 entity1.attributes.get('school_code')]
        codes2 = [entity2.attributes.get('code'), entity2.attributes.get('udise_code'),
                 entity2.attributes.get('school_code')]
        
        # Remove None values
        codes1 = [c for c in codes1 if c]
        codes2 = [c for c in codes2 if c]
        
        for code1 in codes1:
            for code2 in codes2:
                if str(code1) == str(code2) and len(str(code1)) > 4:
                    return EntityLink(
                        entity1_id=entity1.entity_id,
                        entity2_id=entity2.entity_id,
                        entity1_dataset=entity1.dataset,
                        entity2_dataset=entity2.dataset,
                        link_type='same_entity',
                        confidence_score=0.9,
                        bridge_field='code_match',
                        evidence={'matching_codes': [code1, code2]}
                    )
        
        return None
    
    def _geographic_strategy(self, entity1: Entity, entity2: Entity) -> Optional[EntityLink]:
        """Link entities based on geographic relationships"""
        district1 = entity1.attributes.get('district')
        district2 = entity2.attributes.get('district')
        
        if district1 and district2 and district1.lower() == district2.lower():
            # Same district - potential relationship
            confidence = 0.7 if entity1.entity_type == entity2.entity_type else 0.6
            
            return EntityLink(
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                entity1_dataset=entity1.dataset,
                entity2_dataset=entity2.dataset,
                link_type='geographic',
                confidence_score=confidence,
                bridge_field='district',
                evidence={'shared_district': district1}
            )
        
        return None
    
    def _temporal_strategy(self, entity1: Entity, entity2: Entity) -> Optional[EntityLink]:
        """Link entities based on temporal relationships"""
        year1 = entity1.attributes.get('year')
        year2 = entity2.attributes.get('year')
        
        if year1 and year2 and abs(int(year1) - int(year2)) <= 1:
            # Same or consecutive years
            return EntityLink(
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                entity1_dataset=entity1.dataset,
                entity2_dataset=entity2.dataset,
                link_type='temporal',
                confidence_score=0.65,
                bridge_field='year',
                evidence={'years': [year1, year2]}
            )
        
        return None
    
    def _semantic_strategy(self, entity1: Entity, entity2: Entity) -> Optional[EntityLink]:
        """Advanced semantic matching using text analysis"""
        # This would typically use LLM embeddings or NLP models
        # For now, implement rule-based semantic matching
        
        semantic_rules = {
            ('school', 'policy'): self._school_policy_semantic,
            ('district', 'policy'): self._district_policy_semantic,
            ('school', 'district'): self._school_district_semantic
        }
        
        entity_types = tuple(sorted([entity1.entity_type, entity2.entity_type]))
        if entity_types in semantic_rules:
            return semantic_rules[entity_types](entity1, entity2)
        
        return None
    
    def _school_policy_semantic(self, entity1: Entity, entity2: Entity) -> Optional[EntityLink]:
        """Link schools to relevant policies"""
        school = entity1 if entity1.entity_type == 'school' else entity2
        policy = entity2 if entity1.entity_type == 'school' else entity1
        
        # Check if policy mentions school's district
        school_district = school.attributes.get('district', '').lower()
        policy_text = str(policy.attributes).lower()
        
        if school_district and school_district in policy_text:
            return EntityLink(
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                entity1_dataset=entity1.dataset,
                entity2_dataset=entity2.dataset,
                link_type='policy_impact',
                confidence_score=0.7,
                bridge_field='district_policy',
                evidence={'district': school_district, 'policy_mentions_district': True}
            )
        
        return None
    
    def _district_policy_semantic(self, entity1: Entity, entity2: Entity) -> Optional[EntityLink]:
        """Link districts to policies that affect them"""
        district = entity1 if entity1.entity_type == 'district' else entity2
        policy = entity2 if entity1.entity_type == 'district' else entity1
        
        district_name = district.canonical_name.lower()
        policy_context = str(policy.attributes).lower()
        
        if district_name in policy_context:
            return EntityLink(
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                entity1_dataset=entity1.dataset,
                entity2_dataset=entity2.dataset,
                link_type='policy_jurisdiction',
                confidence_score=0.8,
                bridge_field='district_mention',
                evidence={'district': district_name, 'mentioned_in_policy': True}
            )
        
        return None
    
    def _school_district_semantic(self, entity1: Entity, entity2: Entity) -> Optional[EntityLink]:
        """Link schools to their districts"""
        school = entity1 if entity1.entity_type == 'school' else entity2
        district = entity2 if entity1.entity_type == 'school' else entity1
        
        school_district = school.attributes.get('district', '').lower()
        district_name = district.canonical_name.lower()
        
        if school_district == district_name:
            return EntityLink(
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                entity1_dataset=entity1.dataset,
                entity2_dataset=entity2.dataset,
                link_type='hierarchical',
                confidence_score=0.9,
                bridge_field='administrative_hierarchy',
                evidence={'school_district': school_district, 'district_name': district_name}
            )
        
        return None
    
    def _merge_entities(self, existing: Entity, new: Entity) -> None:
        """Merge information from two entities"""
        # Merge attributes
        existing.attributes.update(new.attributes)
        
        # Add new aliases
        for alias in new.aliases:
            if alias not in existing.aliases:
                existing.aliases.append(alias)
    
    def get_entity_graph(self) -> Dict[str, Any]:
        """Generate a graph representation of entities and links"""
        nodes = []
        edges = []
        
        # Add entity nodes
        for entity in self.entities.values():
            nodes.append({
                'id': f"{entity.dataset}_{entity.entity_id}",
                'label': entity.canonical_name,
                'type': entity.entity_type,
                'dataset': entity.dataset,
                'attributes': entity.attributes
            })
        
        # Add links as edges
        for link in self.entity_links:
            edges.append({
                'source': f"{link.entity1_dataset}_{link.entity1_id}",
                'target': f"{link.entity2_dataset}_{link.entity2_id}",
                'type': link.link_type,
                'weight': link.confidence_score,
                'bridge_field': link.bridge_field,
                'evidence': link.evidence
            })
        
        return {'nodes': nodes, 'edges': edges}
    
    def export_links_for_neo4j(self) -> List[Dict[str, Any]]:
        """Export links in format suitable for Neo4j import"""
        neo4j_links = []
        
        for link in self.entity_links:
            neo4j_links.append({
                'entity1_id': link.entity1_id,
                'entity2_id': link.entity2_id,
                'entity1_dataset': link.entity1_dataset,
                'entity2_dataset': link.entity2_dataset,
                'relationship_type': link.link_type.upper(),
                'confidence': link.confidence_score,
                'bridge_field': link.bridge_field,
                'evidence': json.dumps(link.evidence)
            })
        
        return neo4j_links

# Usage example
if __name__ == "__main__":
    # Test with sample data
    resolver = EntityResolver()
    
    # Sample facts (would normally come from pipeline)
    sample_facts = [
        {
            'fact_id': 'SCERT_DOCUMENT_123_Krishna_2023_schools',
            'district': 'Krishna',
            'source_document': 'SCERT_DOCUMENT_123',
            'year': 2023,
            'metadata': {
                'school_codes': ['28193290120', '2500031172'],
                'sample_data': ['SRI GOWTHAMI EM SCHOOL THALARISINGI']
            }
        }
    ]
    
    resolver.add_entities_from_facts(sample_facts)
    links = resolver.resolve_entities()
    
    print(f"Found {len(resolver.entities)} entities")
    print(f"Created {len(links)} links")