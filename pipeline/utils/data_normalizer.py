#!/usr/bin/env python3
"""
Advanced Data Normalizer for AP Education Statistics
Handles district-year-metric normalization with fuzzy matching and validation
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

class DataNormalizer:
    """Advanced normalizer for education statistics with fuzzy matching"""
    
    def __init__(self):
        # Canonical district names (13 districts in AP)
        self.canonical_districts = {
            'anantapur': 'Anantapur',
            'ananthpur': 'Anantapur',
            'anantpur': 'Anantapur',
            'chittoor': 'Chittoor',
            'chitoor': 'Chittoor',
            'chitor': 'Chittoor',
            'east godavari': 'East Godavari',
            'eastgodavari': 'East Godavari',
            'e.godavari': 'East Godavari',
            'e godavari': 'East Godavari',
            'guntur': 'Guntur',
            'guntoor': 'Guntur',
            'kadapa': 'Kadapa',
            'cuddapah': 'Kadapa',
            'cudapah': 'Kadapa',
            'krishna': 'Krishna',
            'kurnool': 'Kurnool',
            'kurnul': 'Kurnool',
            'nellore': 'Nellore',
            'nelore': 'Nellore',
            'prakasam': 'Prakasam',
            'prakasham': 'Prakasam',
            'srikakulam': 'Srikakulam',
            'shrikakulam': 'Srikakulam',
            'visakhapatnam': 'Visakhapatnam',
            'vishakhapatnam': 'Visakhapatnam',
            'vizag': 'Visakhapatnam',
            'vizianagaram': 'Vizianagaram',
            'vijayanagaram': 'Vizianagaram',
            'west godavari': 'West Godavari',
            'westgodavari': 'West Godavari',
            'w.godavari': 'West Godavari',
            'w godavari': 'West Godavari'
        }
        
        # Canonical education indicators
        self.canonical_indicators = {
            # Enrollment indicators
            'enrollment': 'Total Enrollment',
            'enrolment': 'Total Enrollment',
            'total enrollment': 'Total Enrollment',
            'total enrolment': 'Total Enrollment',
            'boys enrollment': 'Boys Enrollment',
            'boys enrolment': 'Boys Enrollment',
            'girls enrollment': 'Girls Enrollment',
            'girls enrolment': 'Girls Enrollment',
            'primary enrollment': 'Primary Enrollment',
            'upper primary enrollment': 'Upper Primary Enrollment',
            'secondary enrollment': 'Secondary Enrollment',
            'higher secondary enrollment': 'Higher Secondary Enrollment',
            
            # Educational efficiency indicators
            'ger': 'Gross Enrollment Ratio',
            'gross enrollment ratio': 'Gross Enrollment Ratio',
            'ner': 'Net Enrollment Ratio',
            'net enrollment ratio': 'Net Enrollment Ratio',
            'gpi': 'Gender Parity Index',
            'gender parity index': 'Gender Parity Index',
            'dropout rate': 'Dropout Rate',
            'drop out rate': 'Dropout Rate',
            'retention rate': 'Retention Rate',
            'transition rate': 'Transition Rate',
            
            # Infrastructure indicators
            'schools': 'Number of Schools',
            'no of schools': 'Number of Schools',
            'number of schools': 'Number of Schools',
            'teachers': 'Number of Teachers',
            'no of teachers': 'Number of Teachers',
            'number of teachers': 'Number of Teachers',
            'ptr': 'Pupil Teacher Ratio',
            'pupil teacher ratio': 'Pupil Teacher Ratio',
            'classrooms': 'Number of Classrooms',
            'toilets': 'Toilet Facilities',
            'drinking water': 'Drinking Water Facilities',
            'electricity': 'Electricity Connection',
            'computers': 'Computer Facilities',
            'library': 'Library Facilities',
            
            # Quality indicators
            'learning outcomes': 'Learning Outcomes',
            'achievement': 'Student Achievement',
            'assessment scores': 'Assessment Scores',
            'competency': 'Competency Levels'
        }
        
        # Category mappings
        self.category_mappings = {
            # Social categories
            'sc': 'SC',
            'scheduled caste': 'SC',
            'st': 'ST',
            'scheduled tribe': 'ST',
            'obc': 'OBC',
            'other backward class': 'OBC',
            'general': 'General',
            'unreserved': 'General',
            
            # Gender categories
            'boys': 'Boys',
            'male': 'Boys',
            'girls': 'Girls',
            'female': 'Girls',
            
            # Location categories
            'rural': 'Rural',
            'urban': 'Urban',
            
            # Education levels
            'primary': 'Primary',
            'upper primary': 'Upper Primary',
            'elementary': 'Elementary',
            'secondary': 'Secondary',
            'higher secondary': 'Higher Secondary'
        }
        
        # Unit mappings
        self.unit_mappings = {
            'percentage': '%',
            'percent': '%',
            'pct': '%',
            'ratio': 'ratio',
            'number': 'count',
            'count': 'count',
            'crore': 'crore',
            'lakh': 'lakh',
            'thousand': 'thousand',
            'rs': 'rupees',
            'rupees': 'rupees'
        }
        
        # Year patterns
        self.year_patterns = [
            r'(\d{4})-(\d{2})',      # 2022-23
            r'(\d{4})-(\d{4})',      # 2022-2023
            r'(\d{4})',              # 2022
            r'AY\s*(\d{4})-(\d{2})', # AY 2022-23
            r'FY\s*(\d{4})-(\d{2})'  # FY 2022-23
        ]
        
    def normalize_extracted_data(self, extracted_data: List[Dict[str, Any]]) -> List[NormalizedFact]:
        """Normalize all extracted data into standard facts"""
        logger.info(f"🔧 Normalizing {len(extracted_data)} extracted items")
        
        normalized_facts = []
        normalization_stats = {
            'total_items': len(extracted_data),
            'facts_created': 0,
            'districts_normalized': 0,
            'indicators_normalized': 0,
            'years_extracted': 0,
            'validation_failed': 0
        }
        
        fact_counter = 0
        
        for item in extracted_data:
            try:
                # Process different types of extracted items
                item_type = item.get('extraction_method', 'unknown')
                
                if item_type in ['camelot_stream', 'ocr_tesseract']:
                    # Table data
                    facts = self._normalize_table_data(item, fact_counter)
                elif item_type == 'pymupdf_text':
                    # Text data
                    facts = self._normalize_text_data(item, fact_counter)
                else:
                    continue
                
                # Validate and add facts
                for fact in facts:
                    if self._validate_fact(fact):
                        normalized_facts.append(fact)
                        fact_counter += 1
                        normalization_stats['facts_created'] += 1
                    else:
                        normalization_stats['validation_failed'] += 1
                        
            except Exception as e:
                logger.error(f"❌ Failed to normalize item: {e}")
                continue
        
        logger.info(f"📊 Normalization complete:")
        logger.info(f"   - Total items processed: {normalization_stats['total_items']}")
        logger.info(f"   - Facts created: {normalization_stats['facts_created']}")
        logger.info(f"   - Validation failures: {normalization_stats['validation_failed']}")
        
        return normalized_facts
    
    def _normalize_table_data(self, table_item: Dict[str, Any], fact_counter: int) -> List[NormalizedFact]:
        """Normalize tabular data into facts"""
        facts = []
        
        try:
            headers = table_item.get('headers', [])
            rows = table_item.get('rows', [])
            
            if not headers or not rows:
                return facts
            
            # Identify district column
            district_col = self._find_district_column(headers)
            if district_col is None:
                return facts
            
            # Identify metric columns
            metric_columns = self._find_metric_columns(headers)
            
            # Extract year from document or table
            year = self._extract_year_from_item(table_item)
            
            for row_idx, row in enumerate(rows):
                if len(row) <= district_col:
                    continue
                
                # Extract district
                raw_district = str(row[district_col]).strip()
                normalized_district = self._normalize_district(raw_district)
                
                if not normalized_district:
                    continue
                
                # Extract metrics from this row
                for col_idx, header in enumerate(headers):
                    if col_idx == district_col or col_idx >= len(row):
                        continue
                    
                    value_str = str(row[col_idx]).strip()
                    if not value_str or value_str.lower() in ['nan', 'null', '']:
                        continue
                    
                    # Try to parse value
                    value, unit = self._parse_value(value_str)
                    if value is None:
                        continue
                    
                    # Normalize indicator
                    normalized_indicator = self._normalize_indicator(header)
                    if not normalized_indicator:
                        continue
                    
                    # Extract category from header or context
                    category = self._extract_category(header)
                    
                    # Create fact
                    fact_id = f"fact_{fact_counter:06d}_{row_idx:03d}_{col_idx:02d}"
                    fact = NormalizedFact(
                        fact_id=fact_id,
                        district=normalized_district,
                        indicator=normalized_indicator,
                        category=category,
                        year=year,
                        value=value,
                        unit=unit,
                        source_document=table_item.get('doc_id', 'unknown'),
                        source_page=table_item.get('page', 1),
                        confidence_score=self._calculate_fact_confidence(
                            normalized_district, normalized_indicator, value, year
                        ),
                        metadata={
                            'extraction_method': table_item.get('extraction_method'),
                            'table_id': table_item.get('table_id'),
                            'original_header': header,
                            'original_value': value_str,
                            'row_index': row_idx,
                            'col_index': col_idx
                        }
                    )
                    
                    facts.append(fact)
        
        except Exception as e:
            logger.error(f"❌ Failed to normalize table data: {e}")
        
        return facts
    
    def _normalize_text_data(self, text_item: Dict[str, Any], fact_counter: int) -> List[NormalizedFact]:
        """Normalize text data into facts"""
        facts = []
        
        try:
            text = text_item.get('text', '')
            if not text:
                return facts
            
            # Look for statistical patterns in text
            patterns = [
                # "District: Value" patterns
                r'([A-Za-z\s]+?):\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|ratio)?',
                # "District has Value" patterns  
                r'([A-Za-z\s]+?)\s+(?:has|shows|records?)\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|ratio)?',
                # Enrollment patterns
                r'enrollment\s+in\s+([A-Za-z\s]+?)\s+(?:is|was)\s+(\d+(?:,\d{3})*(?:\.\d+)?)',
                # Statistical mentions
                r'([A-Za-z\s]+?)\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s+(schools|teachers|students)'
            ]
            
            year = self._extract_year_from_item(text_item)
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    raw_entity = match.group(1).strip()
                    value_str = match.group(2)
                    unit_str = match.group(3) if len(match.groups()) > 2 else ''
                    
                    # Try to identify if entity is a district
                    normalized_district = self._normalize_district(raw_entity)
                    if not normalized_district:
                        continue
                    
                    # Parse value
                    value, unit = self._parse_value(value_str + ' ' + (unit_str or ''))
                    if value is None:
                        continue
                    
                    # Infer indicator from context
                    indicator = self._infer_indicator_from_context(text, match.start(), match.end())
                    if not indicator:
                        continue
                    
                    # Create fact
                    fact_id = f"fact_{fact_counter:06d}_text_{len(facts):03d}"
                    fact = NormalizedFact(
                        fact_id=fact_id,
                        district=normalized_district,
                        indicator=indicator,
                        category=None,
                        year=year,
                        value=value,
                        unit=unit,
                        source_document=text_item.get('doc_id', 'unknown'),
                        source_page=text_item.get('page', 1),
                        confidence_score=0.6,  # Lower confidence for text extraction
                        metadata={
                            'extraction_method': text_item.get('extraction_method'),
                            'text_id': text_item.get('text_id'),
                            'original_text': match.group(0),
                            'pattern_used': pattern
                        }
                    )
                    
                    facts.append(fact)
        
        except Exception as e:
            logger.error(f"❌ Failed to normalize text data: {e}")
        
        return facts
    
    def _find_district_column(self, headers: List[str]) -> Optional[int]:
        """Find the column containing district names"""
        for i, header in enumerate(headers):
            header_lower = str(header).lower()
            if any(word in header_lower for word in ['district', 'dist', 'area', 'region']):
                return i
        return None
    
    def _find_metric_columns(self, headers: List[str]) -> List[int]:
        """Find columns containing metrics"""
        metric_columns = []
        
        for i, header in enumerate(headers):
            header_lower = str(header).lower()
            if any(indicator in header_lower for indicator in self.canonical_indicators.keys()):
                metric_columns.append(i)
        
        return metric_columns
    
    def _normalize_district(self, raw_district: str) -> Optional[str]:
        """Normalize district name using fuzzy matching"""
        if not raw_district or len(raw_district.strip()) < 3:
            return None
        
        raw_lower = raw_district.lower().strip()
        
        # Direct mapping
        if raw_lower in self.canonical_districts:
            return self.canonical_districts[raw_lower]
        
        # Fuzzy matching
        canonical_values = list(set(self.canonical_districts.values()))
        best_match, score = process.extractOne(raw_district, canonical_values, scorer=fuzz.ratio)
        
        if score >= 80:  # High confidence threshold
            return best_match
        
        # Try partial matching for multi-word districts
        for canonical_key, canonical_value in self.canonical_districts.items():
            if fuzz.partial_ratio(raw_lower, canonical_key) >= 85:
                return canonical_value
        
        return None
    
    def _normalize_indicator(self, raw_indicator: str) -> Optional[str]:
        """Normalize education indicator"""
        if not raw_indicator:
            return None
        
        raw_lower = raw_indicator.lower().strip()
        
        # Direct mapping
        if raw_lower in self.canonical_indicators:
            return self.canonical_indicators[raw_lower]
        
        # Fuzzy matching with lower threshold for indicators
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
            
        except ValueError:
            return None, ''
    
    def _extract_year_from_item(self, item: Dict[str, Any]) -> int:
        """Extract year from item data"""
        # Check explicit year field
        if 'year' in item:
            try:
                return int(item['year'])
            except (ValueError, TypeError):
                pass
        
        # Check document ID
        doc_id = item.get('doc_id', '')
        year = self._extract_year_from_text(doc_id)
        if year:
            return year
        
        # Check content/text if available
        content = item.get('text', '') or item.get('content', '')
        year = self._extract_year_from_text(content)
        if year:
            return year
        
        # Default to current year
        return datetime.now().year
    
    def _extract_year_from_text(self, text: str) -> Optional[int]:
        """Extract year from text using patterns"""
        if not text:
            return None
        
        for pattern in self.year_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) >= 2:
                    # Academic year format (2022-23)
                    year1 = int(match.group(1))
                    return year1
                else:
                    # Single year format (2022)
                    return int(match.group(1))
        
        return None
    
    def _infer_indicator_from_context(self, text: str, start: int, end: int) -> Optional[str]:
        """Infer indicator from surrounding context"""
        # Get context around the match
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end].lower()
        
        # Look for indicator keywords in context
        for key, value in self.canonical_indicators.items():
            if key in context:
                return value
        
        # Default indicators based on common patterns
        if any(word in context for word in ['school', 'institution']):
            return 'Number of Schools'
        elif any(word in context for word in ['teacher', 'staff']):
            return 'Number of Teachers'
        elif any(word in context for word in ['student', 'enrollment', 'enrolment']):
            return 'Total Enrollment'
        
        return None
    
    def _calculate_fact_confidence(self, district: str, indicator: str, value: float, year: int) -> float:
        """Calculate confidence score for a normalized fact"""
        score = 0.5  # Base score
        
        # District confidence
        if district and district != 'Unknown':
            score += 0.2
        
        # Indicator confidence
        if indicator and indicator != 'Unknown':
            score += 0.2
        
        # Value reasonableness
        if value is not None and value >= 0:
            score += 0.1
            
            # Check if value is in reasonable range for education metrics
            if indicator and any(word in indicator for word in ['Ratio', 'Rate', '%']):
                if 0 <= value <= 100:
                    score += 0.1
            elif value < 1000000:  # Not unreasonably large
                score += 0.1
        
        # Year reasonableness
        current_year = datetime.now().year
        if 2000 <= year <= current_year:
            score += 0.1
        
        return min(score, 1.0)
    
    def _validate_fact(self, fact: NormalizedFact) -> bool:
        """Validate a normalized fact"""
        try:
            # Basic validation
            if not fact.district or not fact.indicator:
                return False
            
            if fact.value is None or fact.value < 0:
                return False
            
            if fact.year < 2000 or fact.year > datetime.now().year + 1:
                return False
            
            if fact.confidence_score < 0.3:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Fact validation error: {e}")
            return False
    
    def save_normalized_facts(self, facts: List[NormalizedFact], output_file: str):
        """Save normalized facts to JSON file"""
        try:
            # Convert facts to dictionaries
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
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(facts_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved {len(facts)} normalized facts to {output_file}")
            
            # Log summary statistics
            self._log_normalization_summary(facts)
            
        except Exception as e:
            logger.error(f"❌ Failed to save normalized facts: {e}")
    
    def _log_normalization_summary(self, facts: List[NormalizedFact]):
        """Log summary statistics of normalized facts"""
        
        # Count by district
        district_counts = {}
        for fact in facts:
            district_counts[fact.district] = district_counts.get(fact.district, 0) + 1
        
        # Count by indicator
        indicator_counts = {}
        for fact in facts:
            indicator_counts[fact.indicator] = indicator_counts.get(fact.indicator, 0) + 1
        
        # Count by year
        year_counts = {}
        for fact in facts:
            year_counts[fact.year] = year_counts.get(fact.year, 0) + 1
        
        logger.info(f"📊 Normalization Summary:")
        logger.info(f"   - Total facts: {len(facts)}")
        logger.info(f"   - Unique districts: {len(district_counts)}")
        logger.info(f"   - Unique indicators: {len(indicator_counts)}")
        logger.info(f"   - Year range: {min(year_counts.keys())}-{max(year_counts.keys())}")
        
        # Top districts
        top_districts = sorted(district_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"   - Top districts: {', '.join(f'{d}({c})' for d, c in top_districts)}")
        
        # Top indicators
        top_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"   - Top indicators: {', '.join(f'{i}({c})' for i, c in top_indicators)}")


def main():
    """Test the data normalizer"""
    
    # Sample data
    sample_data = [
        {
            'doc_id': 'UDISE_2022',
            'extraction_method': 'camelot_stream',
            'headers': ['District', 'Total Enrollment', 'Boys', 'Girls'],
            'rows': [
                ['Anantapur', '125000', '62000', '63000'],
                ['Chittoor', '138000', '69000', '69000'],
                ['Guntur', '156000', '78000', '78000']
            ],
            'year': '2022'
        }
    ]
    
    normalizer = DataNormalizer()
    facts = normalizer.normalize_extracted_data(sample_data)
    
    print(f"✅ Created {len(facts)} normalized facts")
    for fact in facts[:3]:  # Show first 3
        print(f"  - {fact.district}: {fact.indicator} = {fact.value} ({fact.year})")


if __name__ == "__main__":
    main()