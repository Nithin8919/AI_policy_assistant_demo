#!/usr/bin/env python3
"""
Enhanced Data Normalizer for AP Education Statistics
Handles application lists, statistical data, and creates aggregated facts for bridge tables
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import json
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)

@dataclass
class NormalizedFact:
    """Standardized education fact with consistent dimensions"""
    fact_id: str
    district: str
    indicator: str
    category: Optional[str]  # SC/ST/OBC/General, Boys/Girls, Rural/Urban
    year: int
    value: float
    unit: str
    source_document: str
    source_page: int
    confidence_score: float
    metadata: Dict[str, Any]

class EnhancedDataNormalizer:
    """Enhanced normalizer that handles both application lists and statistical data"""
    
    def __init__(self):
        # Canonical district names (13 districts in AP)
        self.canonical_districts = {
            'anantapur': 'Anantapur', 'ananthpur': 'Anantapur', 'anantpur': 'Anantapur',
            'chittoor': 'Chittoor', 'chitoor': 'Chittoor', 'chitor': 'Chittoor',
            'east godavari': 'East Godavari', 'eastgodavari': 'East Godavari', 
            'e.godavari': 'East Godavari', 'e godavari': 'East Godavari',
            'guntur': 'Guntur', 'guntoor': 'Guntur',
            'krishna': 'Krishna', 'kistna': 'Krishna',
            'kurnool': 'Kurnool', 'kurnol': 'Kurnool',
            'nellore': 'Nellore', 'nelor': 'Nellore',
            'prakasam': 'Prakasam', 'prakash': 'Prakasam',
            'srikakulam': 'Srikakulam', 'srikakula': 'Srikakulam',
            'visakhapatnam': 'Visakhapatnam', 'vishakhapatnam': 'Visakhapatnam',
            'vizag': 'Visakhapatnam', 'visag': 'Visakhapatnam',
            'vizianagaram': 'Vizianagaram', 'vijayanagaram': 'Vizianagaram',
            'west godavari': 'West Godavari', 'westgodavari': 'West Godavari',
            'w.godavari': 'West Godavari', 'w godavari': 'West Godavari',
            'kadapa': 'Kadapa', 'cuddapah': 'Kadapa', 'cudapa': 'Kadapa'
        }
        
        # Canonical indicators with multiple variants
        self.canonical_indicators = {
            'enrollment': 'Enrollment', 'enrolment': 'Enrollment', 'enrolled': 'Enrollment',
            'applications': 'Applications', 'applicants': 'Applications', 'applied': 'Applications',
            'schools': 'Schools', 'institutions': 'Schools', 'school': 'Schools',
            'teachers': 'Teachers', 'faculty': 'Teachers', 'staff': 'Teachers',
            'students': 'Students', 'pupils': 'Students', 'children': 'Students',
            'dropout': 'Dropout Rate', 'dropouts': 'Dropout Rate',
            'ger': 'GER', 'gross enrollment ratio': 'GER', 'gross enrolment ratio': 'GER',
            'ner': 'NER', 'net enrollment ratio': 'NER', 'net enrolment ratio': 'NER',
            'ptr': 'PTR', 'pupil teacher ratio': 'PTR', 'student teacher ratio': 'PTR',
            'literacy': 'Literacy Rate', 'literate': 'Literacy Rate',
            'pass rate': 'Pass Rate', 'passing rate': 'Pass Rate',
            'examination': 'Examination Results', 'exam': 'Examination Results',
            'budget': 'Budget Allocation', 'funds': 'Budget Allocation',
            'infrastructure': 'Infrastructure', 'facilities': 'Infrastructure'
        }
        
        # Category mappings
        self.category_mappings = {
            'sc': 'SC', 'scheduled caste': 'SC', 'scheduled castes': 'SC',
            'st': 'ST', 'scheduled tribe': 'ST', 'scheduled tribes': 'ST',
            'obc': 'OBC', 'other backward class': 'OBC', 'other backward classes': 'OBC',
            'general': 'General', 'open': 'General',
            'boys': 'Boys', 'male': 'Boys', 'men': 'Boys',
            'girls': 'Girls', 'female': 'Girls', 'women': 'Girls',
            'rural': 'Rural', 'village': 'Rural', 'villages': 'Rural',
            'urban': 'Urban', 'city': 'Urban', 'cities': 'Urban', 'town': 'Urban'
        }
        
        # Unit mappings
        self.unit_mappings = {
            'percent': 'percentage', '%': 'percentage', 'percentage': 'percentage',
            'ratio': 'ratio', 'rate': 'rate',
            'count': 'count', 'number': 'count', 'total': 'count',
            'lakh': 'lakh', 'lakhs': 'lakh', 'crore': 'crore', 'crores': 'crore',
            'rupees': 'rupees', 'rs': 'rupees', 'inr': 'rupees'
        }
        
        # Academic year patterns (prioritize these)
        self.year_patterns = [
            r'\b(20[1-2][0-9])[-–—]\d{2}\b',  # 2020-21, 2021-22 format
            r'\b(20[1-2][0-9])\b',  # 2020, 2021 etc (valid education years)
            r'\bYear[:\s]+(20[1-2][0-9])\b',  # Year: 2020
            r'\b(20[0-1][0-9])\b'  # 2000-2019 (broader range)
        ]
        
        # Text patterns for extracting data
        self.district_patterns = [
            r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:District|Mandal|Revenue|Division)',
            r'District[:\s]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
            r'\b(Anantapur|Chittoor|East Godavari|Guntur|Krishna|Kurnool|Nellore|Prakasam|Srikakulam|Visakhapatnam|Vizianagaram|West Godavari|Kadapa)\b'
        ]
        
    def normalize_extracted_data(self, extracted_data: List[Dict[str, Any]]) -> List[NormalizedFact]:
        """Main normalization function that handles different data types"""
        logger.info(f"🔧 Enhanced normalization of {len(extracted_data)} items")
        
        all_facts = []
        
        # Separate data by type
        table_items = []
        text_items = []
        
        for item in extracted_data:
            if self._is_table_data(item):
                table_items.append(item)
            else:
                text_items.append(item)
        
        logger.info(f"📊 Processing {len(table_items)} table items and {len(text_items)} text items")
        
        # Process table data
        if table_items:
            table_facts = self._normalize_table_data(table_items)
            all_facts.extend(table_facts)
            logger.info(f"✅ Extracted {len(table_facts)} facts from tables")
        
        # Process text data (including application lists)
        if text_items:
            text_facts = self._normalize_text_data(text_items)
            all_facts.extend(text_facts)
            logger.info(f"✅ Extracted {len(text_facts)} facts from text")
        
        # Create aggregated facts for bridge tables
        aggregated_facts = self._create_aggregated_facts(all_facts)
        all_facts.extend(aggregated_facts)
        logger.info(f"✅ Created {len(aggregated_facts)} aggregated facts for bridge tables")
        
        logger.info(f"📊 Total normalized facts: {len(all_facts)}")
        return all_facts
    
    def _is_table_data(self, item: Dict[str, Any]) -> bool:
        """Check if item contains table data"""
        return (
            item.get('extraction_method') in ['camelot_stream', 'camelot_lattice', 'ocr_tesseract'] or
            'headers' in item or 'rows' in item or
            item.get('table_id') is not None
        )
    
    def _normalize_table_data(self, table_items: List[Dict[str, Any]]) -> List[NormalizedFact]:
        """Enhanced table data normalization"""
        facts = []
        
        for item in table_items:
            try:
                # Extract basic metadata
                year = self._extract_year_safe(item)
                source_doc = item.get('doc_id', item.get('source_document', 'unknown'))
                source_page = item.get('page_number', 1)
                
                # Try different table processing approaches
                item_facts = []
                
                # Approach 1: Structured headers and rows
                if 'headers' in item and 'rows' in item:
                    item_facts.extend(self._process_structured_table(item, year, source_doc, source_page))
                
                # Approach 2: Generic column analysis
                if not item_facts and 'text' in item:
                    item_facts.extend(self._process_text_table(item, year, source_doc, source_page))
                
                facts.extend(item_facts)
                
            except Exception as e:
                logger.warning(f"Failed to process table item: {e}")
                continue
        
        return facts
    
    def _normalize_text_data(self, text_items: List[Dict[str, Any]]) -> List[NormalizedFact]:
        """Enhanced text data normalization including application lists"""
        facts = []
        
        # Group by document for aggregation
        doc_groups = {}
        for item in text_items:
            doc_id = item.get('doc_id', item.get('source_document', 'unknown'))
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(item)
        
        for doc_id, doc_items in doc_groups.items():
            try:
                # Determine if this is application list data
                sample_text = ' '.join([item.get('text', '')[:200] for item in doc_items[:3]])
                
                if self._is_application_list_data(sample_text):
                    doc_facts = self._process_application_list(doc_items, doc_id)
                else:
                    doc_facts = self._process_statistical_text(doc_items, doc_id)
                
                facts.extend(doc_facts)
                
            except Exception as e:
                logger.warning(f"Failed to process text for document {doc_id}: {e}")
                continue
        
        return facts
    
    def _is_application_list_data(self, text: str) -> bool:
        """Detect if text contains application/enrollment lists"""
        indicators = [
            'application id', 'applicant name', 'parent', 'guardian',
            'school code', 'admission', 'student id', 'roll number',
            'sl.no', 's.no', 'serial number'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for indicator in indicators if indicator in text_lower)
        return matches >= 2
    
    def _process_application_list(self, items: List[Dict[str, Any]], doc_id: str) -> List[NormalizedFact]:
        """Process application/enrollment list data into aggregated facts"""
        facts = []
        
        # Extract year from document context
        year = self._extract_year_from_items(items)
        
        # Aggregate data by district
        district_stats = {}
        
        for item in items:
            text = item.get('text', '')
            if not text or len(text.strip()) < 10:
                continue
            
            # Extract district from text
            district = self._extract_district_from_text(text)
            if not district:
                continue
            
            # Initialize district stats
            if district not in district_stats:
                district_stats[district] = {
                    'applications': 0,
                    'schools_mentioned': set(),
                    'sample_data': []
                }
            
            # Count applications/entries
            district_stats[district]['applications'] += 1
            
            # Extract school information if available
            school_match = re.search(r'(\d{8,})', text)  # School codes are typically 8+ digits
            if school_match:
                district_stats[district]['schools_mentioned'].add(school_match.group(1))
            
            # Store sample for metadata
            if len(district_stats[district]['sample_data']) < 3:
                district_stats[district]['sample_data'].append(text[:100])
        
        # Create facts from aggregated data
        for district, stats in district_stats.items():
            # Applications fact
            if stats['applications'] > 0:
                fact = NormalizedFact(
                    fact_id=f"{doc_id}_{district}_{year}_applications",
                    district=district,
                    indicator="Applications",
                    category=None,
                    year=year,
                    value=float(stats['applications']),
                    unit="count",
                    source_document=doc_id,
                    source_page=1,
                    confidence_score=0.8,
                    metadata={
                        'aggregated_from': 'application_lists',
                        'schools_count': len(stats['schools_mentioned']),
                        'sample_data': stats['sample_data']
                    }
                )
                
                if self._validate_fact(fact):
                    facts.append(fact)
            
            # Schools fact (if we have unique school codes)
            if len(stats['schools_mentioned']) > 0:
                fact = NormalizedFact(
                    fact_id=f"{doc_id}_{district}_{year}_schools",
                    district=district,
                    indicator="Schools",
                    category=None,
                    year=year,
                    value=float(len(stats['schools_mentioned'])),
                    unit="count",
                    source_document=doc_id,
                    source_page=1,
                    confidence_score=0.7,
                    metadata={
                        'aggregated_from': 'application_lists',
                        'school_codes': list(stats['schools_mentioned'])
                    }
                )
                
                if self._validate_fact(fact):
                    facts.append(fact)
        
        return facts
    
    def _process_statistical_text(self, items: List[Dict[str, Any]], doc_id: str) -> List[NormalizedFact]:
        """Process statistical text data"""
        facts = []
        
        for item in items:
            text = item.get('text', '')
            if not text or len(text.strip()) < 20:
                continue
            
            # Extract year
            year = self._extract_year_safe(item)
            
            # Look for statistical patterns
            # Pattern: District: Value, Indicator: Value, etc.
            stat_patterns = [
                r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)[:\s]+(\d+(?:,\d{3})*(?:\.\d+)?)',
                r'(\w+(?:\s\w+)*)[:\s]+(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|ratio)',
            ]
            
            for pattern in stat_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    label = match.group(1).strip()
                    value_str = match.group(2).replace(',', '')
                    
                    # Try to parse as district or indicator
                    district = self._normalize_district(label)
                    indicator = self._normalize_indicator(label)
                    
                    if district or indicator:
                        try:
                            value = float(value_str)
                            unit = 'percentage' if len(match.groups()) > 2 else 'count'
                            
                            fact = NormalizedFact(
                                fact_id=f"{doc_id}_{district or 'unknown'}_{indicator or 'unknown'}_{year}",
                                district=district or 'Unknown',
                                indicator=indicator or 'Unknown',
                                category=None,
                                year=year,
                                value=value,
                                unit=unit,
                                source_document=doc_id,
                                source_page=item.get('page_number', 1),
                                confidence_score=0.6,
                                metadata={'extracted_from': 'statistical_text'}
                            )
                            
                            if self._validate_fact(fact):
                                facts.append(fact)
                        
                        except (ValueError, TypeError):
                            continue
        
        return facts
    
    def _process_structured_table(self, item: Dict[str, Any], year: int, source_doc: str, source_page: int) -> List[NormalizedFact]:
        """Process tables with headers and rows"""
        facts = []
        
        headers = item.get('headers', [])
        rows = item.get('rows', [])
        
        if not headers or not rows:
            return facts
        
        # Find district column
        district_col = self._find_district_column_enhanced(headers)
        
        # Process each row
        for row_idx, row in enumerate(rows):
            if not row or len(row) != len(headers):
                continue
            
            try:
                # Extract district
                district = None
                if district_col is not None and district_col < len(row):
                    district = self._normalize_district(str(row[district_col]))
                
                # Extract values from other columns
                for col_idx, (header, cell) in enumerate(zip(headers, row)):
                    if col_idx == district_col:
                        continue
                    
                    # Try to parse value
                    value, unit = self._parse_value_safe(str(cell))
                    if value is None:
                        continue
                    
                    # Normalize indicator from header
                    indicator = self._normalize_indicator(str(header))
                    category = self._extract_category(str(header))
                    
                    if indicator and district:
                        fact = NormalizedFact(
                            fact_id=f"{source_doc}_{district}_{indicator}_{year}_{row_idx}_{col_idx}",
                            district=district,
                            indicator=indicator,
                            category=category,
                            year=year,
                            value=value,
                            unit=unit,
                            source_document=source_doc,
                            source_page=source_page,
                            confidence_score=0.9,
                            metadata={'table_row': row_idx, 'table_col': col_idx}
                        )
                        
                        if self._validate_fact(fact):
                            facts.append(fact)
            
            except Exception as e:
                logger.warning(f"Failed to process table row {row_idx}: {e}")
                continue
        
        return facts
    
    def _find_district_column_enhanced(self, headers: List[str]) -> Optional[int]:
        """Enhanced district column detection"""
        for i, header in enumerate(headers):
            header_lower = str(header).lower()
            
            # Direct matches
            if any(word in header_lower for word in ['district', 'dist', 'area', 'region']):
                return i
            
            # Check if header contains district names
            for district_name in self.canonical_districts.values():
                if district_name.lower() in header_lower:
                    return i
        
        # Check if any column contains mostly district names
        if hasattr(self, '_current_table_rows'):
            for col_idx in range(len(headers)):
                district_matches = 0
                total_cells = 0
                
                for row in self._current_table_rows[:10]:  # Check first 10 rows
                    if col_idx < len(row):
                        cell_value = str(row[col_idx]).strip()
                        if cell_value and len(cell_value) > 2:
                            total_cells += 1
                            if self._normalize_district(cell_value):
                                district_matches += 1
                
                # If >50% of cells in this column are districts
                if total_cells > 0 and district_matches / total_cells > 0.5:
                    return col_idx
        
        return None
    
    def _create_aggregated_facts(self, facts: List[NormalizedFact]) -> List[NormalizedFact]:
        """Create aggregated facts for bridge tables"""
        aggregated = []
        
        # Group facts by district-year
        district_year_groups = {}
        for fact in facts:
            key = f"{fact.district}_{fact.year}"
            if key not in district_year_groups:
                district_year_groups[key] = []
            district_year_groups[key].append(fact)
        
        # Create district-year summary facts
        for key, group_facts in district_year_groups.items():
            district, year = key.split('_')
            year = int(year)
            
            # Total applications
            app_facts = [f for f in group_facts if f.indicator == 'Applications']
            if app_facts:
                total_apps = sum(f.value for f in app_facts)
                
                summary_fact = NormalizedFact(
                    fact_id=f"SUMMARY_{district}_{year}_total_applications",
                    district=district,
                    indicator="Total Applications",
                    category=None,
                    year=year,
                    value=total_apps,
                    unit="count",
                    source_document="AGGREGATED",
                    source_page=0,
                    confidence_score=0.95,
                    metadata={
                        'aggregated_from': len(app_facts),
                        'bridge_table_key': f"{district}_{year}",
                        'summary_type': 'district_year_total'
                    }
                )
                aggregated.append(summary_fact)
        
        return aggregated
    
    # Utility methods
    def _extract_year_safe(self, item: Dict[str, Any]) -> int:
        """Safely extract year with validation"""
        # Try document context first
        year = self._extract_year_from_text(item.get('text', ''))
        
        if not year or year < 2000 or year > datetime.now().year + 1:
            # Default to recent year for education data
            year = 2023
        
        return year
    
    def _extract_year_from_items(self, items: List[Dict[str, Any]]) -> int:
        """Extract year from multiple items"""
        for item in items:
            year = self._extract_year_safe(item)
            if year and 2010 <= year <= datetime.now().year:
                return year
        return 2023
    
    def _extract_year_from_text(self, text: str) -> Optional[int]:
        """Extract year from text using enhanced patterns"""
        if not text:
            return None
        
        for pattern in self.year_patterns:
            match = re.search(pattern, text)
            if match:
                year = int(match.group(1))
                # Validate year range (reject document ID years like 1760)
                if 2000 <= year <= datetime.now().year + 1:
                    return year
        
        return None
    
    def _extract_district_from_text(self, text: str) -> Optional[str]:
        """Extract district name from text"""
        if not text:
            return None
        
        # Try exact matches first
        text_lower = text.lower()
        for key, canonical in self.canonical_districts.items():
            if key in text_lower:
                return canonical
        
        # Try pattern matching
        for pattern in self.district_patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1)
                normalized = self._normalize_district(candidate)
                if normalized:
                    return normalized
        
        return None
    
    def _normalize_district(self, raw_district: str) -> Optional[str]:
        """Normalize district name using fuzzy matching"""
        if not raw_district:
            return None
        
        raw_lower = raw_district.lower().strip()
        
        # Exact match
        if raw_lower in self.canonical_districts:
            return self.canonical_districts[raw_lower]
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for key, value in self.canonical_districts.items():
            score = fuzz.partial_ratio(raw_lower, key)
            if score > best_score and score >= 80:  # High threshold for district names
                best_score = score
                best_match = value
        
        return best_match
    
    def _normalize_indicator(self, raw_indicator: str) -> Optional[str]:
        """Normalize indicator name"""
        if not raw_indicator:
            return None
        
        raw_lower = raw_indicator.lower().strip()
        
        # Exact match
        if raw_lower in self.canonical_indicators:
            return self.canonical_indicators[raw_lower]
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for key, value in self.canonical_indicators.items():
            score = fuzz.partial_ratio(raw_lower, key)
            if score > best_score and score >= 70:
                best_score = score
                best_match = value
        
        return best_match
    
    def _extract_category(self, header: str) -> Optional[str]:
        """Extract category from header text"""
        if not header:
            return None
        
        header_lower = header.lower()
        
        for key, value in self.category_mappings.items():
            if key in header_lower:
                return value
        
        return None
    
    def _parse_value_safe(self, value_str: str) -> Tuple[Optional[float], str]:
        """Safely parse value and unit from string"""
        try:
            return self._parse_value(value_str)
        except:
            return None, 'count'
    
    def _parse_value(self, value_str: str) -> Tuple[Optional[float], str]:
        """Parse value and unit from string"""
        if not value_str:
            return None, ''
        
        # Clean the string
        cleaned = re.sub(r'[^\d.,\-+%a-zA-Z\s]', '', str(value_str))
        cleaned = cleaned.strip()
        
        # Extract numeric part
        numeric_match = re.search(r'([\d,]+(?:\.\d+)?)', cleaned)
        if not numeric_match:
            return None, ''
        
        try:
            # Parse number (handle commas)
            numeric_str = numeric_match.group(1).replace(',', '')
            value = float(numeric_str)
            
            # Extract unit
            unit_part = cleaned[numeric_match.end():].strip().lower()
            
            # Normalize unit
            unit = 'count'  # default
            for key, normalized_unit in self.unit_mappings.items():
                if key in unit_part:
                    unit = normalized_unit
                    break
            
            return value, unit
        
        except (ValueError, TypeError):
            return None, ''
    
    def _validate_fact(self, fact: NormalizedFact) -> bool:
        """Validate a normalized fact"""
        try:
            # Required fields
            if not all([fact.fact_id, fact.district, fact.indicator]):
                return False
            
            # Year validation
            if fact.year < 2000 or fact.year > datetime.now().year + 1:
                return False
            
            # Value validation
            if fact.value is None or fact.value < 0:
                return False
            
            # District validation
            if fact.district not in self.canonical_districts.values() and fact.district != 'Unknown':
                return False
            
            return True
            
        except Exception:
            return False
    
    def save_normalized_facts(self, facts: List[NormalizedFact], output_file: str):
        """Save normalized facts to JSON file"""
        facts_data = []
        for fact in facts:
            fact_dict = {
                'fact_id': fact.fact_id,
                'district': fact.district,
                'indicator': fact.indicator,
                'category': fact.category,
                'year': fact.year,
                'value': fact.value,
                'unit': fact.unit,
                'source_document': fact.source_document,
                'source_page': fact.source_page,
                'confidence_score': fact.confidence_score,
                'metadata': fact.metadata
            }
            facts_data.append(fact_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(facts_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(facts_data)} normalized facts to {output_file}")