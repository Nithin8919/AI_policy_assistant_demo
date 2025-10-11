"""
NLP Pipeline for Policy Documents - NER, RE, EL, ER
"""
import spacy
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Optional imports
try:
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification, 
        pipeline, AutoModelForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
import hashlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Entity structure"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    entity_id: str
    context: str = ""

@dataclass
class Relation:
    """Relation structure"""
    head: Entity
    tail: Entity
    relation_type: str
    confidence: float
    context: str
    relation_id: str

class PolicyNLPProcessor:
    """NLP processor for policy documents with NER, RE, EL, ER"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"spaCy model {model_name} not found. Please install it first.")
            raise
        
        # Initialize legal NER model (InLegalBERT)
        self.legal_ner_model = None
        self.legal_ner_tokenizer = None
        self._setup_legal_ner()
        
        # Initialize relation extraction model
        self.relation_model = None
        self._setup_relation_extraction()
        
        # Initialize embedding model for entity linking
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Entity and relation patterns
        self._setup_patterns()
    
    def _setup_legal_ner(self):
        """Setup legal NER model (InLegalBERT)"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Try to load InLegalBERT model
                model_name = "law-ai/InLegalBERT"
                self.legal_ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.legal_ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
                
                logger.info("Loaded InLegalBERT model for legal NER")
            except Exception as e:
                logger.warning(f"Failed to load InLegalBERT: {e}. Using spaCy fallback.")
                self.legal_ner_model = None
        else:
            logger.info("Transformers not available, using spaCy only")
            self.legal_ner_model = None
    
    def _setup_relation_extraction(self):
        """Setup relation extraction model"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a general relation extraction model
                # In production, you might want to fine-tune on legal data
                self.relation_model = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",
                    return_all_scores=True
                )
                logger.info("Loaded relation extraction model")
            except Exception as e:
                logger.warning(f"Failed to load relation extraction model: {e}")
                self.relation_model = None
        else:
            logger.info("Transformers not available, using pattern-based relation extraction")
            self.relation_model = None
    
    def _setup_patterns(self):
        """Setup regex patterns for entity extraction"""
        self.patterns = {
            'GO_NUMBER': [
                r'G\.O\.Ms\.No\.\s*(\d+)',
                r'G\.O\.Ms\.\s*(\d+)',
                r'GO\s*(\d+)',
                r'Government\s+Order\s+(\d+)'
            ],
            'CIRCULAR_NUMBER': [
                r'Circular\s+No\.?\s*(\d+)',
                r'Circ\.\s*(\d+)',
                r'CSE\s*(\d+)',
                r'Notification\s+No\.?\s*(\d+)'
            ],
            'DATE': [
                r'\d{1,2}-\d{1,2}-\d{4}',
                r'\d{1,2}/\d{1,2}/\d{4}',
                r'\d{4}-\d{1,2}-\d{1,2}',
                r'\d{1,2}\s+\w+\s+\d{4}'
            ],
            'AMOUNT': [
                r'Rs\.?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*rupees'
            ],
            'PERCENTAGE': [
                r'(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*percent'
            ],
            'SCHOOL_TYPE': [
                r'Primary\s+School',
                r'High\s+School',
                r'Secondary\s+School',
                r'Elementary\s+School',
                r'Government\s+School',
                r'Private\s+School',
                r'Aided\s+School'
            ]
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities using multiple methods
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract entities using spaCy
        spacy_entities = self._extract_spacy_entities(text)
        entities.extend(spacy_entities)
        
        # Extract entities using legal NER model
        if self.legal_ner_model:
            legal_entities = self._extract_legal_entities(text)
            entities.extend(legal_entities)
        
        # Extract entities using regex patterns
        pattern_entities = self._extract_pattern_entities(text)
        entities.extend(pattern_entities)
        
        # Remove duplicates and merge overlapping entities
        entities = self._merge_entities(entities)
        
        logger.info(f"Extracted {len(entities)} entities")
        return entities
    
    def _extract_spacy_entities(self, text: str) -> List[Entity]:
        """Extract entities using spaCy"""
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,  # spaCy doesn't provide confidence scores
                entity_id=self._generate_entity_id(ent.text, ent.label_),
                context=text[max(0, ent.start_char-50):ent.end_char+50]
            )
            entities.append(entity)
        
        return entities
    
    def _extract_legal_entities(self, text: str) -> List[Entity]:
        """Extract entities using legal NER model"""
        entities = []
        
        try:
            # Tokenize text
            tokens = self.legal_ner_tokenizer.tokenize(text)
            inputs = self.legal_ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.legal_ner_model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
            
            # Convert predictions to entities
            current_entity = None
            for i, (token, pred) in enumerate(zip(tokens, predictions[0])):
                if pred != 0:  # Not O (outside) tag
                    label = self.legal_ner_model.config.id2label[pred.item()]
                    
                    if current_entity is None or current_entity.label != label:
                        if current_entity:
                            entities.append(current_entity)
                        
                        current_entity = Entity(
                            text=token,
                            label=label,
                            start=i,
                            end=i+1,
                            confidence=0.8,  # Placeholder confidence
                            entity_id=self._generate_entity_id(token, label),
                            context=""
                        )
                    else:
                        current_entity.text += " " + token
                        current_entity.end = i + 1
            
            if current_entity:
                entities.append(current_entity)
                
        except Exception as e:
            logger.error(f"Legal NER extraction failed: {e}")
        
        return entities
    
    def _extract_pattern_entities(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = Entity(
                        text=match.group(0),
                        label=pattern_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,  # High confidence for regex matches
                        entity_id=self._generate_entity_id(match.group(0), pattern_type),
                        context=text[max(0, match.start()-50):match.end()+50]
                    )
                    entities.append(entity)
        
        return entities
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        Extract relations between entities
        
        Args:
            text: Input text
            entities: List of extracted entities
            
        Returns:
            List of extracted relations
        """
        relations = []
        
        # Extract relations using pattern-based approach
        pattern_relations = self._extract_pattern_relations(text, entities)
        relations.extend(pattern_relations)
        
        # Extract relations using ML model
        if self.relation_model:
            ml_relations = self._extract_ml_relations(text, entities)
            relations.extend(ml_relations)
        
        logger.info(f"Extracted {len(relations)} relations")
        return relations
    
    def _extract_pattern_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using pattern-based approach"""
        relations = []
        
        # Define relation patterns
        relation_patterns = {
            'AMENDS': [
                r'(\w+)\s+amends?\s+(\w+)',
                r'(\w+)\s+modifies?\s+(\w+)',
                r'(\w+)\s+changes?\s+(\w+)'
            ],
            'IMPLEMENTS': [
                r'(\w+)\s+implements?\s+(\w+)',
                r'(\w+)\s+enforces?\s+(\w+)',
                r'(\w+)\s+executes?\s+(\w+)'
            ],
            'GOVERNED_BY': [
                r'(\w+)\s+governed\s+by\s+(\w+)',
                r'(\w+)\s+regulated\s+by\s+(\w+)',
                r'(\w+)\s+controlled\s+by\s+(\w+)'
            ],
            'APPLIES_TO': [
                r'(\w+)\s+applies?\s+to\s+(\w+)',
                r'(\w+)\s+affects?\s+(\w+)',
                r'(\w+)\s+concerns?\s+(\w+)'
            ]
        }
        
        for relation_type, patterns in relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    head_text = match.group(1)
                    tail_text = match.group(2)
                    
                    # Find corresponding entities
                    head_entity = self._find_entity_by_text(entities, head_text)
                    tail_entity = self._find_entity_by_text(entities, tail_text)
                    
                    if head_entity and tail_entity:
                        relation = Relation(
                            head=head_entity,
                            tail=tail_entity,
                            relation_type=relation_type,
                            confidence=0.8,
                            context=match.group(0),
                            relation_id=self._generate_relation_id(head_entity.entity_id, tail_entity.entity_id, relation_type)
                        )
                        relations.append(relation)
        
        return relations
    
    def _extract_ml_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using ML model"""
        relations = []
        
        try:
            # Create entity pairs
            for i, head in enumerate(entities):
                for tail in entities[i+1:]:
                    # Create context around both entities
                    start = min(head.start, tail.start)
                    end = max(head.end, tail.end)
                    context = text[max(0, start-100):end+100]
                    
                    # Use ML model to classify relation
                    if self.relation_model:
                        # This is a simplified approach - in practice, you'd need a proper relation extraction model
                        relation_score = 0.5  # Placeholder
                        
                        if relation_score > 0.7:
                            relation = Relation(
                                head=head,
                                tail=tail,
                                relation_type='RELATED',
                                confidence=relation_score,
                                context=context,
                                relation_id=self._generate_relation_id(head.entity_id, tail.entity_id, 'RELATED')
                            )
                            relations.append(relation)
        
        except Exception as e:
            logger.error(f"ML relation extraction failed: {e}")
        
        return relations
    
    def entity_linking(self, entities: List[Entity]) -> List[Entity]:
        """
        Link entities to canonical forms
        
        Args:
            entities: List of entities
            
        Returns:
            List of entities with linked canonical forms
        """
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            if entity.label not in entities_by_type:
                entities_by_type[entity.label] = []
            entities_by_type[entity.label].append(entity)
        
        # Link entities within each type
        linked_entities = []
        for entity_type, type_entities in entities_by_type.items():
            linked_type_entities = self._link_entities_of_type(type_entities)
            linked_entities.extend(linked_type_entities)
        
        logger.info(f"Linked {len(linked_entities)} entities")
        return linked_entities
    
    def _link_entities_of_type(self, entities: List[Entity]) -> List[Entity]:
        """Link entities of the same type"""
        if len(entities) <= 1:
            return entities
        
        # Generate embeddings for entity texts
        texts = [entity.text for entity in entities]
        embeddings = self.embedding_model.encode(texts)
        
        # Cluster similar entities
        similarity_matrix = cosine_similarity(embeddings)
        
        # Use DBSCAN to cluster similar entities
        clustering = DBSCAN(eps=0.3, min_samples=1, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Create canonical forms for each cluster
        linked_entities = []
        for cluster_id in set(cluster_labels):
            cluster_entities = [entities[i] for i in range(len(entities)) if cluster_labels[i] == cluster_id]
            
            if len(cluster_entities) == 1:
                linked_entities.append(cluster_entities[0])
            else:
                # Choose the most frequent or longest entity as canonical form
                canonical_entity = max(cluster_entities, key=lambda e: len(e.text))
                
                # Update all entities in cluster to point to canonical form
                for entity in cluster_entities:
                    entity.entity_id = canonical_entity.entity_id
                    entity.text = canonical_entity.text
                
                linked_entities.append(canonical_entity)
        
        return linked_entities
    
    def entity_resolution(self, entities: List[Entity]) -> List[Entity]:
        """
        Resolve entity references and disambiguate
        
        Args:
            entities: List of entities
            
        Returns:
            List of resolved entities
        """
        resolved_entities = []
        
        # Create entity resolution rules
        resolution_rules = {
            'PERSON': self._resolve_person_entities,
            'ORG': self._resolve_org_entities,
            'GPE': self._resolve_gpe_entities,
            'LAW': self._resolve_law_entities
        }
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            if entity.label not in entities_by_type:
                entities_by_type[entity.label] = []
            entities_by_type[entity.label].append(entity)
        
        # Resolve entities by type
        for entity_type, type_entities in entities_by_type.items():
            if entity_type in resolution_rules:
                resolved_type_entities = resolution_rules[entity_type](type_entities)
            else:
                resolved_type_entities = type_entities
            
            resolved_entities.extend(resolved_type_entities)
        
        logger.info(f"Resolved {len(resolved_entities)} entities")
        return resolved_entities
    
    def _resolve_person_entities(self, entities: List[Entity]) -> List[Entity]:
        """Resolve person entities"""
        # Simple resolution - could be enhanced with external knowledge bases
        return entities
    
    def _resolve_org_entities(self, entities: List[Entity]) -> List[Entity]:
        """Resolve organization entities"""
        # Map common abbreviations to full names
        org_mappings = {
            'AP': 'Andhra Pradesh',
            'GOI': 'Government of India',
            'MHRD': 'Ministry of Human Resource Development',
            'SCERT': 'State Council of Educational Research and Training',
            'CSE': 'Commissioner of School Education'
        }
        
        for entity in entities:
            if entity.text.upper() in org_mappings:
                entity.text = org_mappings[entity.text.upper()]
        
        return entities
    
    def _resolve_gpe_entities(self, entities: List[Entity]) -> List[Entity]:
        """Resolve geopolitical entities"""
        return entities
    
    def _resolve_law_entities(self, entities: List[Entity]) -> List[Entity]:
        """Resolve legal entities"""
        # Map common legal references
        law_mappings = {
            'RTE': 'Right to Education Act',
            'NEP': 'National Education Policy',
            'GO': 'Government Order'
        }
        
        for entity in entities:
            if entity.text.upper() in law_mappings:
                entity.text = law_mappings[entity.text.upper()]
        
        return entities
    
    def _find_entity_by_text(self, entities: List[Entity], text: str) -> Optional[Entity]:
        """Find entity by text (fuzzy matching)"""
        text_lower = text.lower()
        for entity in entities:
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                return entity
        return None
    
    def _generate_entity_id(self, text: str, label: str) -> str:
        """Generate unique entity ID"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{label}_{text_hash}"
    
    def _generate_relation_id(self, head_id: str, tail_id: str, relation_type: str) -> str:
        """Generate unique relation ID"""
        relation_hash = hashlib.md5(f"{head_id}_{tail_id}_{relation_type}".encode()).hexdigest()[:8]
        return f"REL_{relation_hash}"
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities"""
        if not entities:
            return entities
        
        # Sort entities by start position
        entities.sort(key=lambda e: e.start)
        
        merged_entities = []
        current_entity = entities[0]
        
        for next_entity in entities[1:]:
            # Check if entities overlap
            if next_entity.start <= current_entity.end:
                # Merge entities
                if next_entity.end > current_entity.end:
                    current_entity.end = next_entity.end
                    current_entity.text = current_entity.text + " " + next_entity.text
                # Keep the entity with higher confidence
                if next_entity.confidence > current_entity.confidence:
                    current_entity = next_entity
            else:
                merged_entities.append(current_entity)
                current_entity = next_entity
        
        merged_entities.append(current_entity)
        return merged_entities
    
    def process_document(self, text: str, doc_id: str) -> Dict[str, Any]:
        """
        Process a document through the complete NLP pipeline
        
        Args:
            text: Document text
            doc_id: Document identifier
            
        Returns:
            Processing results
        """
        logger.info(f"Processing document: {doc_id}")
        
        try:
            # Extract entities
            entities = self.extract_entities(text)
            
            # Extract relations
            relations = self.extract_relations(text, entities)
            
            # Entity linking
            linked_entities = self.entity_linking(entities)
            
            # Entity resolution
            resolved_entities = self.entity_resolution(linked_entities)
            
            # Prepare results
            results = {
                'doc_id': doc_id,
                'entities': [
                    {
                        'entity_id': e.entity_id,
                        'text': e.text,
                        'label': e.label,
                        'start': e.start,
                        'end': e.end,
                        'confidence': e.confidence,
                        'context': e.context
                    }
                    for e in resolved_entities
                ],
                'relations': [
                    {
                        'relation_id': r.relation_id,
                        'head_entity_id': r.head.entity_id,
                        'tail_entity_id': r.tail.entity_id,
                        'relation_type': r.relation_type,
                        'confidence': r.confidence,
                        'context': r.context
                    }
                    for r in relations
                ],
                'statistics': {
                    'total_entities': len(resolved_entities),
                    'total_relations': len(relations),
                    'entity_types': len(set(e.label for e in resolved_entities)),
                    'relation_types': len(set(r.relation_type for r in relations))
                }
            }
            
            logger.info(f"Processed document {doc_id}: {len(resolved_entities)} entities, {len(relations)} relations")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            return {'doc_id': doc_id, 'error': str(e)}

def main():
    """Test the NLP processor"""
    processor = PolicyNLPProcessor()
    
    # Sample text
    sample_text = """
    Government Order No. 75 of 2021 amends Rule 5 of the Education Act.
    The Right to Education Act applies to all schools in Andhra Pradesh.
    SCERT implements the National Education Policy guidelines.
    """
    
    # Process the text
    results = processor.process_document(sample_text, "test_doc")
    
    print("NLP Processing Results:")
    print(f"Entities: {len(results.get('entities', []))}")
    print(f"Relations: {len(results.get('relations', []))}")
    
    for entity in results.get('entities', [])[:5]:
        print(f"- {entity['text']} ({entity['label']})")
    
    for relation in results.get('relations', [])[:3]:
        print(f"- {relation['head_entity_id']} -> {relation['tail_entity_id']} ({relation['relation_type']})")

if __name__ == "__main__":
    main()
