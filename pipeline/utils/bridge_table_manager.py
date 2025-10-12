#!/usr/bin/env python3
"""
Bridge Table Manager for AP Policy Co-Pilot
Creates and manages first-level connection tables for fast fact retrieval
"""
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BridgeConnection:
    """Represents a bridge connection between facts"""
    source_fact_id: str
    target_fact_id: str
    connection_type: str  # 'district_year', 'indicator_temporal', 'policy_hierarchy'
    strength: float  # 0.0 to 1.0
    metadata: Dict[str, Any]

@dataclass
class BridgeTable:
    """Represents a bridge table with aggregated facts and connections"""
    bridge_id: str
    bridge_type: str
    key_dimensions: Dict[str, Any]  # e.g., {'district': 'Anantapur', 'year': 2023}
    connected_facts: List[str]  # fact_ids
    summary_stats: Dict[str, Any]
    indicators_covered: Set[str]
    connections: List[BridgeConnection]
    metadata: Dict[str, Any]

class BridgeTableManager:
    """Manages bridge tables for first-level connections and fast retrieval"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.bridge_dir = self.data_dir / "bridge_tables"
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize bridge tables storage
        self.district_year_bridges: Dict[str, BridgeTable] = {}
        self.indicator_temporal_bridges: Dict[str, BridgeTable] = {}
        self.policy_hierarchy_bridges: Dict[str, BridgeTable] = {}
        self.cross_district_bridges: Dict[str, BridgeTable] = {}
        
        # Connection strength thresholds
        self.connection_thresholds = {
            'district_year': 0.8,      # Same district and year
            'indicator_temporal': 0.7,  # Same indicator, adjacent years
            'policy_hierarchy': 0.6,    # Related policy documents
            'cross_district': 0.5       # Same indicator, same year, different districts
        }
        
    def create_bridge_tables_from_facts(self, facts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Create all bridge tables from normalized facts"""
        logger.info(f"🌉 Creating bridge tables from {len(facts)} facts")
        
        # Group facts for bridge creation
        district_year_groups = self._group_facts_by_district_year(facts)
        indicator_temporal_groups = self._group_facts_by_indicator_temporal(facts)
        policy_groups = self._group_facts_by_policy_hierarchy(facts)
        
        # Create bridge tables
        stats = {
            'district_year_bridges': self._create_district_year_bridges(district_year_groups),
            'indicator_temporal_bridges': self._create_indicator_temporal_bridges(indicator_temporal_groups),
            'policy_hierarchy_bridges': self._create_policy_hierarchy_bridges(policy_groups),
            'cross_district_bridges': self._create_cross_district_bridges(facts)
        }
        
        # Create connections between bridge tables
        self._create_bridge_connections()
        
        # Save all bridge tables
        self._save_bridge_tables()
        
        logger.info(f"✅ Created {sum(stats.values())} bridge tables total")
        return stats
    
    def _group_facts_by_district_year(self, facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group facts by district-year combination"""
        groups = defaultdict(list)
        
        for fact in facts:
            district = fact.get('district', 'Unknown')
            year = fact.get('year', 2023)
            key = f"{district}_{year}"
            groups[key].append(fact)
        
        return dict(groups)
    
    def _group_facts_by_indicator_temporal(self, facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group facts by indicator across time"""
        groups = defaultdict(list)
        
        for fact in facts:
            indicator = fact.get('indicator', 'Unknown')
            district = fact.get('district', 'Unknown')
            key = f"{indicator}_{district}"
            groups[key].append(fact)
        
        return dict(groups)
    
    def _group_facts_by_policy_hierarchy(self, facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group facts by policy document hierarchy"""
        groups = defaultdict(list)
        
        for fact in facts:
            source_doc = fact.get('source_document', 'Unknown')
            # Extract document type for grouping
            doc_type = self._extract_document_type(source_doc)
            groups[doc_type].append(fact)
        
        return dict(groups)
    
    def _create_district_year_bridges(self, groups: Dict[str, List[Dict[str, Any]]]) -> int:
        """Create district-year bridge tables"""
        created_count = 0
        
        for key, facts in groups.items():
            if len(facts) < 2:  # Need at least 2 facts for a meaningful bridge
                continue
            
            district, year = key.split('_')
            year = int(year)
            
            # Calculate summary statistics
            summary_stats = self._calculate_district_year_stats(facts)
            indicators = set(fact.get('indicator', 'Unknown') for fact in facts)
            
            # Create connections between facts
            connections = self._create_fact_connections(facts, 'district_year')
            
            # Create bridge table
            bridge = BridgeTable(
                bridge_id=f"DY_{key}",
                bridge_type="district_year",
                key_dimensions={'district': district, 'year': year},
                connected_facts=[fact['fact_id'] for fact in facts],
                summary_stats=summary_stats,
                indicators_covered=indicators,
                connections=connections,
                metadata={
                    'created_at': datetime.now().isoformat(),
                    'fact_count': len(facts),
                    'connection_count': len(connections)
                }
            )
            
            self.district_year_bridges[key] = bridge
            created_count += 1
        
        logger.info(f"📊 Created {created_count} district-year bridges")
        return created_count
    
    def _create_indicator_temporal_bridges(self, groups: Dict[str, List[Dict[str, Any]]]) -> int:
        """Create indicator temporal bridge tables"""
        created_count = 0
        
        for key, facts in groups.items():
            if len(facts) < 2:
                continue
            
            indicator, district = key.split('_', 1)
            
            # Sort facts by year for temporal analysis
            facts_sorted = sorted(facts, key=lambda x: x.get('year', 2023))
            
            # Calculate temporal statistics
            summary_stats = self._calculate_temporal_stats(facts_sorted)
            years = sorted(set(fact.get('year', 2023) for fact in facts))
            
            # Create temporal connections
            connections = self._create_temporal_connections(facts_sorted)
            
            # Create bridge table
            bridge = BridgeTable(
                bridge_id=f"IT_{key}",
                bridge_type="indicator_temporal",
                key_dimensions={'indicator': indicator, 'district': district},
                connected_facts=[fact['fact_id'] for fact in facts],
                summary_stats=summary_stats,
                indicators_covered={indicator},
                connections=connections,
                metadata={
                    'created_at': datetime.now().isoformat(),
                    'year_range': [min(years), max(years)] if years else [None, None],
                    'temporal_span': max(years) - min(years) + 1 if years else 0
                }
            )
            
            self.indicator_temporal_bridges[key] = bridge
            created_count += 1
        
        logger.info(f"📈 Created {created_count} indicator temporal bridges")
        return created_count
    
    def _create_policy_hierarchy_bridges(self, groups: Dict[str, List[Dict[str, Any]]]) -> int:
        """Create policy hierarchy bridge tables"""
        created_count = 0
        
        for doc_type, facts in groups.items():
            if len(facts) < 3:  # Need more facts for policy grouping
                continue
            
            # Group by policy relevance
            policy_groups = self._group_by_policy_relevance(facts)
            
            for policy_key, policy_facts in policy_groups.items():
                if len(policy_facts) < 2:
                    continue
                
                # Calculate policy statistics
                summary_stats = self._calculate_policy_stats(policy_facts)
                indicators = set(fact.get('indicator', 'Unknown') for fact in policy_facts)
                
                # Create policy connections
                connections = self._create_policy_connections(policy_facts)
                
                # Create bridge table
                bridge = BridgeTable(
                    bridge_id=f"PH_{doc_type}_{policy_key}",
                    bridge_type="policy_hierarchy",
                    key_dimensions={'document_type': doc_type, 'policy_area': policy_key},
                    connected_facts=[fact['fact_id'] for fact in policy_facts],
                    summary_stats=summary_stats,
                    indicators_covered=indicators,
                    connections=connections,
                    metadata={
                        'created_at': datetime.now().isoformat(),
                        'document_count': len(set(fact.get('source_document') for fact in policy_facts))
                    }
                )
                
                self.policy_hierarchy_bridges[f"{doc_type}_{policy_key}"] = bridge
                created_count += 1
        
        logger.info(f"⚖️ Created {created_count} policy hierarchy bridges")
        return created_count
    
    def _create_cross_district_bridges(self, facts: List[Dict[str, Any]]) -> int:
        """Create cross-district comparison bridges"""
        created_count = 0
        
        # Group by indicator and year
        indicator_year_groups = defaultdict(list)
        for fact in facts:
            indicator = fact.get('indicator', 'Unknown')
            year = fact.get('year', 2023)
            key = f"{indicator}_{year}"
            indicator_year_groups[key].append(fact)
        
        # Create bridges for indicators present in multiple districts
        for key, facts_group in indicator_year_groups.items():
            districts = set(fact.get('district', 'Unknown') for fact in facts_group)
            
            if len(districts) < 3:  # Need at least 3 districts for comparison
                continue
            
            indicator, year = key.split('_')
            year = int(year)
            
            # Calculate cross-district statistics
            summary_stats = self._calculate_cross_district_stats(facts_group)
            
            # Create cross-district connections
            connections = self._create_cross_district_connections(facts_group)
            
            # Create bridge table
            bridge = BridgeTable(
                bridge_id=f"CD_{key}",
                bridge_type="cross_district",
                key_dimensions={'indicator': indicator, 'year': year},
                connected_facts=[fact['fact_id'] for fact in facts_group],
                summary_stats=summary_stats,
                indicators_covered={indicator},
                connections=connections,
                metadata={
                    'created_at': datetime.now().isoformat(),
                    'districts_covered': sorted(districts),
                    'district_count': len(districts)
                }
            )
            
            self.cross_district_bridges[key] = bridge
            created_count += 1
        
        logger.info(f"🗺️ Created {created_count} cross-district bridges")
        return created_count
    
    def _calculate_district_year_stats(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for district-year bridge"""
        values = [fact.get('value', 0) for fact in facts if fact.get('value') is not None]
        
        if not values:
            return {'count': 0}
        
        return {
            'fact_count': len(facts),
            'total_value': sum(values),
            'average_value': sum(values) / len(values),
            'max_value': max(values),
            'min_value': min(values),
            'value_range': max(values) - min(values),
            'indicators': list(set(fact.get('indicator', 'Unknown') for fact in facts)),
            'confidence_avg': sum(fact.get('confidence_score', 0.5) for fact in facts) / len(facts)
        }
    
    def _calculate_temporal_stats(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate temporal statistics for indicator bridges"""
        values = [(fact.get('year', 2023), fact.get('value', 0)) for fact in facts 
                 if fact.get('value') is not None]
        
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend
        years = [v[0] for v in values]
        vals = [v[1] for v in values]
        
        # Simple linear trend
        if len(vals) >= 2:
            trend = (vals[-1] - vals[0]) / (years[-1] - years[0]) if years[-1] != years[0] else 0
        else:
            trend = 0
        
        return {
            'data_points': len(values),
            'year_range': [min(years), max(years)],
            'value_trend': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
            'trend_slope': trend,
            'latest_value': vals[-1] if vals else None,
            'earliest_value': vals[0] if vals else None,
            'average_value': sum(vals) / len(vals) if vals else 0
        }
    
    def _calculate_policy_stats(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate policy-related statistics"""
        documents = set(fact.get('source_document', '') for fact in facts)
        districts = set(fact.get('district', '') for fact in facts)
        years = set(fact.get('year', 2023) for fact in facts)
        
        return {
            'document_count': len(documents),
            'district_coverage': len(districts),
            'temporal_span': len(years),
            'fact_count': len(facts),
            'policy_scope': 'multi_district' if len(districts) > 1 else 'single_district',
            'documents': list(documents)
        }
    
    def _calculate_cross_district_stats(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cross-district comparison statistics"""
        district_values = defaultdict(list)
        
        for fact in facts:
            district = fact.get('district', 'Unknown')
            value = fact.get('value')
            if value is not None:
                district_values[district].append(value)
        
        # Calculate district averages
        district_averages = {}
        for district, values in district_values.items():
            district_averages[district] = sum(values) / len(values)
        
        if not district_averages:
            return {'comparison': 'no_data'}
        
        # Find highest and lowest performing districts
        sorted_districts = sorted(district_averages.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'district_count': len(district_averages),
            'highest_district': sorted_districts[0][0] if sorted_districts else None,
            'highest_value': sorted_districts[0][1] if sorted_districts else None,
            'lowest_district': sorted_districts[-1][0] if sorted_districts else None,
            'lowest_value': sorted_districts[-1][1] if sorted_districts else None,
            'district_averages': district_averages,
            'state_average': sum(district_averages.values()) / len(district_averages),
            'performance_gap': sorted_districts[0][1] - sorted_districts[-1][1] if len(sorted_districts) > 1 else 0
        }
    
    def _create_fact_connections(self, facts: List[Dict[str, Any]], connection_type: str) -> List[BridgeConnection]:
        """Create connections between facts based on type"""
        connections = []
        
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i+1:]:
                strength = self._calculate_connection_strength(fact1, fact2, connection_type)
                
                if strength >= self.connection_thresholds.get(connection_type, 0.5):
                    connection = BridgeConnection(
                        source_fact_id=fact1['fact_id'],
                        target_fact_id=fact2['fact_id'],
                        connection_type=connection_type,
                        strength=strength,
                        metadata={'reason': self._get_connection_reason(fact1, fact2, connection_type)}
                    )
                    connections.append(connection)
        
        return connections
    
    def _create_temporal_connections(self, facts: List[Dict[str, Any]]) -> List[BridgeConnection]:
        """Create temporal connections between facts"""
        connections = []
        
        for i, fact1 in enumerate(facts[:-1]):
            fact2 = facts[i+1]  # Next fact in temporal sequence
            
            year1 = fact1.get('year', 2023)
            year2 = fact2.get('year', 2023)
            
            # Create connection for adjacent years
            if abs(year2 - year1) <= 2:  # Within 2 years
                strength = max(0.5, 1.0 - (abs(year2 - year1) * 0.2))
                
                connection = BridgeConnection(
                    source_fact_id=fact1['fact_id'],
                    target_fact_id=fact2['fact_id'],
                    connection_type='temporal_sequence',
                    strength=strength,
                    metadata={
                        'year_gap': abs(year2 - year1),
                        'trend_direction': 'forward' if year2 > year1 else 'backward'
                    }
                )
                connections.append(connection)
        
        return connections
    
    def _create_policy_connections(self, facts: List[Dict[str, Any]]) -> List[BridgeConnection]:
        """Create policy-based connections"""
        return self._create_fact_connections(facts, 'policy_hierarchy')
    
    def _create_cross_district_connections(self, facts: List[Dict[str, Any]]) -> List[BridgeConnection]:
        """Create cross-district comparison connections"""
        return self._create_fact_connections(facts, 'cross_district')
    
    def _calculate_connection_strength(self, fact1: Dict[str, Any], fact2: Dict[str, Any], connection_type: str) -> float:
        """Calculate connection strength between two facts"""
        if connection_type == 'district_year':
            # Same district and year = strong connection
            same_district = fact1.get('district') == fact2.get('district')
            same_year = fact1.get('year') == fact2.get('year')
            same_indicator = fact1.get('indicator') == fact2.get('indicator')
            
            if same_district and same_year:
                return 0.9 if same_indicator else 0.8
            elif same_district:
                year_diff = abs(fact1.get('year', 2023) - fact2.get('year', 2023))
                return max(0.3, 0.7 - (year_diff * 0.1))
            
            return 0.0
        
        elif connection_type == 'indicator_temporal':
            # Same indicator across time
            same_indicator = fact1.get('indicator') == fact2.get('indicator')
            same_district = fact1.get('district') == fact2.get('district')
            
            if same_indicator and same_district:
                year_diff = abs(fact1.get('year', 2023) - fact2.get('year', 2023))
                return max(0.4, 0.9 - (year_diff * 0.15))
            
            return 0.0
        
        elif connection_type == 'cross_district':
            # Same indicator and year, different districts
            same_indicator = fact1.get('indicator') == fact2.get('indicator')
            same_year = fact1.get('year') == fact2.get('year')
            different_district = fact1.get('district') != fact2.get('district')
            
            if same_indicator and same_year and different_district:
                return 0.7
            
            return 0.0
        
        elif connection_type == 'policy_hierarchy':
            # Same source document or related policy area
            same_doc = fact1.get('source_document') == fact2.get('source_document')
            if same_doc:
                return 0.8
            
            # Check if documents are related (basic heuristic)
            doc1_type = self._extract_document_type(fact1.get('source_document', ''))
            doc2_type = self._extract_document_type(fact2.get('source_document', ''))
            
            if doc1_type == doc2_type:
                return 0.6
            
            return 0.0
        
        return 0.0
    
    def _get_connection_reason(self, fact1: Dict[str, Any], fact2: Dict[str, Any], connection_type: str) -> str:
        """Get human-readable reason for connection"""
        if connection_type == 'district_year':
            return f"Same district ({fact1.get('district')}) and year ({fact1.get('year')})"
        elif connection_type == 'indicator_temporal':
            return f"Same indicator ({fact1.get('indicator')}) across years {fact1.get('year')}-{fact2.get('year')}"
        elif connection_type == 'cross_district':
            return f"Comparison of {fact1.get('indicator')} between {fact1.get('district')} and {fact2.get('district')}"
        elif connection_type == 'policy_hierarchy':
            return f"Related policy documents"
        else:
            return f"Connection type: {connection_type}"
    
    def _extract_document_type(self, doc_name: str) -> str:
        """Extract document type from document name"""
        if not doc_name:
            return 'unknown'
        
        doc_lower = doc_name.lower()
        
        if 'go_' in doc_lower or 'government_order' in doc_lower:
            return 'government_order'
        elif 'cse_' in doc_lower:
            return 'cse_document'
        elif 'scert_' in doc_lower:
            return 'scert_document'
        elif 'act' in doc_lower:
            return 'act'
        elif 'rule' in doc_lower:
            return 'rule'
        else:
            return 'general_document'
    
    def _group_by_policy_relevance(self, facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group facts by policy relevance"""
        groups = defaultdict(list)
        
        for fact in facts:
            # Group by indicator for policy analysis
            indicator = fact.get('indicator', 'Unknown')
            groups[indicator].append(fact)
        
        return dict(groups)
    
    def _create_bridge_connections(self):
        """Create connections between different bridge tables"""
        # This creates meta-connections between bridge tables themselves
        # For now, we'll keep it simple and focus on individual bridge tables
        pass
    
    def _save_bridge_tables(self):
        """Save all bridge tables to disk"""
        # Save district-year bridges
        dy_file = self.bridge_dir / "district_year_bridges.json"
        self._save_bridge_collection(self.district_year_bridges, dy_file)
        
        # Save indicator temporal bridges
        it_file = self.bridge_dir / "indicator_temporal_bridges.json"
        self._save_bridge_collection(self.indicator_temporal_bridges, it_file)
        
        # Save policy hierarchy bridges
        ph_file = self.bridge_dir / "policy_hierarchy_bridges.json"
        self._save_bridge_collection(self.policy_hierarchy_bridges, ph_file)
        
        # Save cross-district bridges
        cd_file = self.bridge_dir / "cross_district_bridges.json"
        self._save_bridge_collection(self.cross_district_bridges, cd_file)
        
        # Save bridge index
        self._save_bridge_index()
        
        logger.info(f"💾 Saved all bridge tables to {self.bridge_dir}")
    
    def _save_bridge_collection(self, bridges: Dict[str, BridgeTable], file_path: Path):
        """Save a collection of bridge tables"""
        bridge_data = {}
        
        for key, bridge in bridges.items():
            bridge_dict = asdict(bridge)
            # Convert sets to lists for JSON serialization
            if isinstance(bridge_dict['indicators_covered'], set):
                bridge_dict['indicators_covered'] = list(bridge_dict['indicators_covered'])
            
            bridge_data[key] = bridge_dict
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(bridge_data, f, indent=2, ensure_ascii=False)
    
    def _save_bridge_index(self):
        """Save bridge table index for fast lookup"""
        index = {
            'created_at': datetime.now().isoformat(),
            'total_bridges': (
                len(self.district_year_bridges) + 
                len(self.indicator_temporal_bridges) + 
                len(self.policy_hierarchy_bridges) + 
                len(self.cross_district_bridges)
            ),
            'bridge_types': {
                'district_year': len(self.district_year_bridges),
                'indicator_temporal': len(self.indicator_temporal_bridges),
                'policy_hierarchy': len(self.policy_hierarchy_bridges),
                'cross_district': len(self.cross_district_bridges)
            },
            'lookup_keys': {
                'district_year': list(self.district_year_bridges.keys()),
                'indicator_temporal': list(self.indicator_temporal_bridges.keys()),
                'policy_hierarchy': list(self.policy_hierarchy_bridges.keys()),
                'cross_district': list(self.cross_district_bridges.keys())
            }
        }
        
        index_file = self.bridge_dir / "bridge_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    
    def get_related_facts(self, query_context: Dict[str, Any], max_results: int = 20) -> List[str]:
        """Get related fact IDs using bridge tables"""
        related_fact_ids = set()
        
        # Extract query dimensions
        district = query_context.get('district')
        year = query_context.get('year')
        indicator = query_context.get('indicator')
        
        # Search district-year bridges
        if district and year:
            key = f"{district}_{year}"
            if key in self.district_year_bridges:
                bridge = self.district_year_bridges[key]
                related_fact_ids.update(bridge.connected_facts)
        
        # Search indicator temporal bridges
        if indicator and district:
            key = f"{indicator}_{district}"
            if key in self.indicator_temporal_bridges:
                bridge = self.indicator_temporal_bridges[key]
                related_fact_ids.update(bridge.connected_facts)
        
        # Search cross-district bridges
        if indicator and year:
            key = f"{indicator}_{year}"
            if key in self.cross_district_bridges:
                bridge = self.cross_district_bridges[key]
                related_fact_ids.update(bridge.connected_facts)
        
        return list(related_fact_ids)[:max_results]
    
    def load_bridge_tables(self):
        """Load bridge tables from disk"""
        try:
            # Load district-year bridges
            dy_file = self.bridge_dir / "district_year_bridges.json"
            if dy_file.exists():
                with open(dy_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, bridge_dict in data.items():
                        # Convert lists back to sets
                        bridge_dict['indicators_covered'] = set(bridge_dict['indicators_covered'])
                        # Convert connections back to BridgeConnection objects
                        connections = []
                        for conn_dict in bridge_dict.get('connections', []):
                            connections.append(BridgeConnection(**conn_dict))
                        bridge_dict['connections'] = connections
                        
                        self.district_year_bridges[key] = BridgeTable(**bridge_dict)
            
            # Similar loading for other bridge types...
            logger.info(f"📥 Loaded bridge tables from {self.bridge_dir}")
            
        except Exception as e:
            logger.error(f"Failed to load bridge tables: {e}")