#!/usr/bin/env python3
"""
Stage 2: Schema Normalization Engine
Normalizes extracted tables into unified dataset with canonical indicators
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re
from difflib import SequenceMatcher

# Fuzzy matching
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)

class SchemaNormalizer:
    """Production-ready schema normalizer for AP education policy data"""
    
    def __init__(self, output_dir: str = "data/normalized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Canonical data dictionary
        self.data_dictionary = self._build_data_dictionary()
        
        # Lookup tables for normalization
        self.indicator_lookup = self._build_indicator_lookup()
        self.district_lookup = self._build_district_lookup()
        self.category_lookup = self._build_category_lookup()
        
        # Validation schemas
        self.validation_schemas = self._build_validation_schemas()
    
    def _build_data_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive data dictionary for AP education indicators"""
        return {
            'GER': {
                'full_name': 'Gross Enrolment Ratio',
                'description': 'Total enrolment in a specific level of education, regardless of age',
                'unit': 'ratio',
                'range': (0, 1.5),
                'category': 'enrolment'
            },
            'NER': {
                'full_name': 'Net Enrolment Ratio',
                'description': 'Enrolment of the official age group in a specific level of education',
                'unit': 'ratio',
                'range': (0, 1),
                'category': 'enrolment'
            },
            'GPI': {
                'full_name': 'Gender Parity Index',
                'description': 'Ratio of female to male enrolment',
                'unit': 'ratio',
                'range': (0, 2),
                'category': 'gender'
            },
            'PTR': {
                'full_name': 'Pupil Teacher Ratio',
                'description': 'Number of pupils per teacher',
                'unit': 'count',
                'range': (1, 100),
                'category': 'teacher'
            },
            'Dropout_Rate': {
                'full_name': 'Dropout Rate',
                'description': 'Percentage of students who leave school before completion',
                'unit': 'percentage',
                'range': (0, 100),
                'category': 'retention'
            },
            'Retention_Rate': {
                'full_name': 'Retention Rate',
                'description': 'Percentage of students who continue to next grade',
                'unit': 'percentage',
                'range': (0, 100),
                'category': 'retention'
            },
            'Enrolment': {
                'full_name': 'Total Enrolment',
                'description': 'Total number of students enrolled',
                'unit': 'count',
                'range': (0, 1000000),
                'category': 'enrolment'
            },
            'Teachers': {
                'full_name': 'Number of Teachers',
                'description': 'Total number of teachers',
                'unit': 'count',
                'range': (0, 100000),
                'category': 'teacher'
            },
            'Schools': {
                'full_name': 'Number of Schools',
                'description': 'Total number of schools',
                'unit': 'count',
                'range': (0, 100000),
                'category': 'infrastructure'
            },
            'Classrooms': {
                'full_name': 'Number of Classrooms',
                'description': 'Total number of classrooms',
                'unit': 'count',
                'range': (0, 500000),
                'category': 'infrastructure'
            }
        }
    
    def _build_indicator_lookup(self) -> Dict[str, str]:
        """Build fuzzy lookup table for indicators"""
        return {
            # GER variations
            'gross enrolment ratio': 'GER',
            'ger': 'GER',
            'gross enrollment ratio': 'GER',
            'enrolment ratio': 'GER',
            'enrollment ratio': 'GER',
            
            # NER variations
            'net enrolment ratio': 'NER',
            'ner': 'NER',
            'net enrollment ratio': 'NER',
            
            # GPI variations
            'gender parity index': 'GPI',
            'gpi': 'GPI',
            'gender parity': 'GPI',
            'female male ratio': 'GPI',
            
            # PTR variations
            'pupil teacher ratio': 'PTR',
            'ptr': 'PTR',
            'student teacher ratio': 'PTR',
            'teacher pupil ratio': 'PTR',
            
            # Dropout variations
            'dropout rate': 'Dropout_Rate',
            'drop out rate': 'Dropout_Rate',
            'drop-out rate': 'Dropout_Rate',
            'discontinuation rate': 'Dropout_Rate',
            
            # Retention variations
            'retention rate': 'Retention_Rate',
            'continuation rate': 'Retention_Rate',
            'survival rate': 'Retention_Rate',
            
            # Enrolment variations
            'enrolment': 'Enrolment',
            'enrollment': 'Enrolment',
            'total enrolment': 'Enrolment',
            'total enrollment': 'Enrolment',
            'student count': 'Enrolment',
            'pupil count': 'Enrolment',
            
            # Teacher variations
            'teachers': 'Teachers',
            'teacher count': 'Teachers',
            'number of teachers': 'Teachers',
            'teaching staff': 'Teachers',
            
            # School variations
            'schools': 'Schools',
            'school count': 'Schools',
            'number of schools': 'Schools',
            'educational institutions': 'Schools',
            
            # Classroom variations
            'classrooms': 'Classrooms',
            'class room': 'Classrooms',
            'class room count': 'Classrooms',
            'number of classrooms': 'Classrooms'
        }
    
    def _build_district_lookup(self) -> Dict[str, str]:
        """Build fuzzy lookup table for districts"""
        ap_districts = [
            'Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Kadapa', 
            'Krishna', 'Kurnool', 'Nellore', 'Prakasam', 'Srikakulam', 
            'Visakhapatnam', 'Vizianagaram', 'West Godavari'
        ]
        
        lookup = {}
        for district in ap_districts:
            # Add exact match
            lookup[district.lower()] = district
            
            # Add common variations
            if 'godavari' in district.lower():
                if 'east' in district.lower():
                    lookup['east godavari'] = district
                    lookup['eg'] = district
                elif 'west' in district.lower():
                    lookup['west godavari'] = district
                    lookup['wg'] = district
            
            # Add abbreviations
            lookup[district[:3].lower()] = district
        
        return lookup
    
    def _build_category_lookup(self) -> Dict[str, str]:
        """Build lookup table for categories"""
        return {
            'boys': 'male',
            'girls': 'female',
            'male': 'male',
            'female': 'female',
            'rural': 'rural',
            'urban': 'urban',
            'government': 'government',
            'private': 'private',
            'aided': 'aided',
            'unaided': 'unaided',
            'total': 'total',
            'all': 'total'
        }
    
    def _build_validation_schemas(self) -> Dict[str, Any]:
        """Build validation schemas for different data types"""
        return {
            'fact_schema': {
                'fact_id': str,
                'indicator': str,
                'category': str,
                'district': str,
                'year': str,
                'value': float,
                'unit': str,
                'source': str,
                'page_ref': int,
                'confidence': float
            },
            'indicator_schema': {
                'indicator': str,
                'full_name': str,
                'description': str,
                'unit': str,
                'category': str
            }
        }
    
    def normalize_extracted_data(self, extracted_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Normalize all extracted data into unified fact table
        
        Args:
            extracted_data: Raw extracted data from Stage 1
            
        Returns:
            List of normalized facts
        """
        logger.info("Starting schema normalization")
        
        normalized_facts = []
        
        for pdf_name, items in extracted_data.items():
            for item in items:
                try:
                    if item.get('extraction_method') in ['camelot_stream', 'ocr_tesseract']:
                        # Process table data
                        facts = self._normalize_table(item, pdf_name)
                        normalized_facts.extend(facts)
                    
                    elif item.get('extraction_method') == 'pymupdf_text':
                        # Process text data
                        facts = self._normalize_text(item, pdf_name)
                        normalized_facts.extend(facts)
                
                except Exception as e:
                    logger.error(f"Failed to normalize item from {pdf_name}: {e}")
                    continue
        
        # Validate normalized facts
        validated_facts = self._validate_normalized_facts(normalized_facts)
        
        # Save normalized data
        self._save_normalized_data(validated_facts)
        
        logger.info(f"Normalized {len(validated_facts)} facts")
        return validated_facts
    
    def _normalize_table(self, table_item: Dict[str, Any], pdf_name: str) -> List[Dict[str, Any]]:
        """Normalize table data into facts"""
        facts = []
        
        try:
            headers = table_item.get('headers', [])
            rows = table_item.get('rows', [])
            
            if not headers or not rows:
                return facts
            
            # Identify key columns
            district_col = self._find_district_column(headers)
            indicator_col = self._find_indicator_column(headers)
            value_cols = self._find_value_columns(headers)
            
            # Process each row
            for row_idx, row in enumerate(rows):
                if len(row) != len(headers):
                    continue
                
                # Extract district
                district = self._normalize_district(row[district_col]) if district_col is not None else 'Unknown'
                
                # Extract indicator
                indicator = self._normalize_indicator(headers[indicator_col]) if indicator_col is not None else 'Unknown'
                
                # Extract values
                for value_col in value_cols:
                    try:
                        value = self._normalize_value(row[value_col])
                        if value is not None:
                            fact = {
                                'fact_id': f"FCT_{table_item['table_id']}_{row_idx}_{value_col}",
                                'indicator': indicator,
                                'category': self._extract_category(headers[value_col]),
                                'district': district,
                                'year': table_item.get('year', 'Unknown'),
                                'value': value,
                                'unit': self._extract_unit(headers[value_col]),
                                'source': table_item.get('source_type', 'Unknown'),
                                'page_ref': table_item.get('page', 0),
                                'confidence': table_item.get('confidence', 0.8),
                                'table_id': table_item['table_id'],
                                'pdf_name': pdf_name,
                                'normalized_at': datetime.now().isoformat()
                            }
                            facts.append(fact)
                    
                    except Exception as e:
                        logger.debug(f"Failed to process value in row {row_idx}, col {value_col}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Table normalization failed: {e}")
        
        return facts
    
    def _normalize_text(self, text_item: Dict[str, Any], pdf_name: str) -> List[Dict[str, Any]]:
        """Normalize text data into facts"""
        facts = []
        
        try:
            text = text_item.get('text', '')
            
            # Look for numeric patterns with context
            numeric_patterns = self._extract_numeric_patterns(text)
            
            for pattern in numeric_patterns:
                fact = {
                    'fact_id': f"FCT_{text_item['text_id']}_{len(facts)}",
                    'indicator': pattern['indicator'],
                    'category': pattern['category'],
                    'district': pattern['district'],
                    'year': text_item.get('year', 'Unknown'),
                    'value': pattern['value'],
                    'unit': pattern['unit'],
                    'source': text_item.get('source_type', 'Unknown'),
                    'page_ref': text_item.get('page', 0),
                    'confidence': pattern['confidence'],
                    'text_id': text_item['text_id'],
                    'pdf_name': pdf_name,
                    'normalized_at': datetime.now().isoformat()
                }
                facts.append(fact)
        
        except Exception as e:
            logger.error(f"Text normalization failed: {e}")
        
        return facts
    
    def _find_district_column(self, headers: List[str]) -> Optional[int]:
        """Find district column index"""
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if any(district.lower() in header_lower for district in self.district_lookup.values()):
                return i
            if 'district' in header_lower or 'mandal' in header_lower:
                return i
        return None
    
    def _find_indicator_column(self, headers: List[str]) -> Optional[int]:
        """Find indicator column index"""
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if any(indicator.lower() in header_lower for indicator in self.indicator_lookup.keys()):
                return i
        return None
    
    def _find_value_columns(self, headers: List[str]) -> List[int]:
        """Find numeric value columns"""
        value_cols = []
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if any(keyword in header_lower for keyword in ['boys', 'girls', 'total', 'count', 'number', 'rate', 'ratio']):
                value_cols.append(i)
        return value_cols
    
    def _normalize_district(self, district_text: str) -> str:
        """Normalize district name using fuzzy matching"""
        if not district_text or pd.isna(district_text):
            return 'Unknown'
        
        district_lower = str(district_text).lower().strip()
        
        # Exact match
        if district_lower in self.district_lookup:
            return self.district_lookup[district_lower]
        
        # Fuzzy match
        best_match = process.extractOne(district_lower, list(self.district_lookup.keys()))
        if best_match and best_match[1] > 80:  # 80% similarity threshold
            return self.district_lookup[best_match[0]]
        
        return 'Unknown'
    
    def _normalize_indicator(self, indicator_text: str) -> str:
        """Normalize indicator name using fuzzy matching"""
        if not indicator_text or pd.isna(indicator_text):
            return 'Unknown'
        
        indicator_lower = str(indicator_text).lower().strip()
        
        # Exact match
        if indicator_lower in self.indicator_lookup:
            return self.indicator_lookup[indicator_lower]
        
        # Fuzzy match
        best_match = process.extractOne(indicator_lower, list(self.indicator_lookup.keys()))
        if best_match and best_match[1] > 70:  # 70% similarity threshold
            return self.indicator_lookup[best_match[0]]
        
        return 'Unknown'
    
    def _normalize_value(self, value_text: Any) -> Optional[float]:
        """Normalize value to float"""
        if pd.isna(value_text):
            return None
        
        try:
            # Convert to string and clean
            value_str = str(value_text).strip()
            
            # Remove common artifacts
            value_str = value_str.replace(',', '').replace('%', '').replace('*', '')
            
            # Extract numeric part
            numeric_match = re.search(r'[\d.]+', value_str)
            if numeric_match:
                return float(numeric_match.group())
            
            return None
        
        except Exception:
            return None
    
    def _extract_category(self, header_text: str) -> str:
        """Extract category from header text"""
        header_lower = header_text.lower()
        
        for category, canonical in self.category_lookup.items():
            if category in header_lower:
                return canonical
        
        return 'total'
    
    def _extract_unit(self, header_text: str) -> str:
        """Extract unit from header text"""
        header_lower = header_text.lower()
        
        if '%' in header_text or 'percentage' in header_lower:
            return 'percentage'
        elif 'ratio' in header_lower:
            return 'ratio'
        elif any(keyword in header_lower for keyword in ['count', 'number', 'total']):
            return 'count'
        else:
            return 'unknown'
    
    def _extract_numeric_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract numeric patterns from text"""
        patterns = []
        
        # Common patterns for education statistics
        patterns_to_find = [
            r'(\d+(?:\.\d+)?)\s*%\s*(?:gross enrolment ratio|ger)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:net enrolment ratio|ner)',
            r'(\d+(?:\.\d+)?)\s*(?:pupil teacher ratio|ptr)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:dropout rate)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:retention rate)'
        ]
        
        for pattern in patterns_to_find:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                value = float(match.group(1))
                indicator = self._extract_indicator_from_pattern(match.group(0))
                
                patterns.append({
                    'indicator': indicator,
                    'category': 'total',
                    'district': 'Unknown',
                    'value': value,
                    'unit': 'percentage' if '%' in match.group(0) else 'ratio',
                    'confidence': 0.6
                })
        
        return patterns
    
    def _extract_indicator_from_pattern(self, pattern_text: str) -> str:
        """Extract indicator from pattern text"""
        pattern_lower = pattern_text.lower()
        
        if 'ger' in pattern_lower or 'gross enrolment' in pattern_lower:
            return 'GER'
        elif 'ner' in pattern_lower or 'net enrolment' in pattern_lower:
            return 'NER'
        elif 'ptr' in pattern_lower or 'pupil teacher' in pattern_lower:
            return 'PTR'
        elif 'dropout' in pattern_lower:
            return 'Dropout_Rate'
        elif 'retention' in pattern_lower:
            return 'Retention_Rate'
        else:
            return 'Unknown'
    
    def _validate_normalized_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate normalized facts"""
        validated_facts = []
        
        for fact in facts:
            try:
                # Check required fields
                required_fields = ['fact_id', 'indicator', 'district', 'year', 'value', 'source']
                if not all(field in fact for field in required_fields):
                    continue
                
                # Validate value range
                indicator = fact['indicator']
                if indicator in self.data_dictionary:
                    expected_range = self.data_dictionary[indicator]['range']
                    if not (expected_range[0] <= fact['value'] <= expected_range[1]):
                        logger.warning(f"Value {fact['value']} out of range for {indicator}")
                        continue
                
                # Validate district
                if fact['district'] == 'Unknown':
                    continue
                
                validated_facts.append(fact)
            
            except Exception as e:
                logger.error(f"Fact validation failed: {e}")
                continue
        
        return validated_facts
    
    def _save_normalized_data(self, facts: List[Dict[str, Any]]):
        """Save normalized data to files"""
        try:
            # Save as JSON
            json_file = self.output_dir / "normalized_facts.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(facts, f, indent=2, ensure_ascii=False)
            
            # Save as CSV
            if facts:
                df = pd.DataFrame(facts)
                csv_file = self.output_dir / "normalized_facts.csv"
                df.to_csv(csv_file, index=False)
            
            # Save data dictionary
            dict_file = self.output_dir / "data_dictionary.json"
            with open(dict_file, 'w', encoding='utf-8') as f:
                json.dump(self.data_dictionary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved normalized data to {self.output_dir}")
        
        except Exception as e:
            logger.error(f"Failed to save normalized data: {e}")
    
    def generate_summary_report(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary report of normalized data"""
        if not facts:
            return {}
        
        df = pd.DataFrame(facts)
        
        summary = {
            'total_facts': len(facts),
            'unique_indicators': df['indicator'].nunique(),
            'unique_districts': df['district'].nunique(),
            'unique_years': df['year'].nunique(),
            'sources': df['source'].value_counts().to_dict(),
            'indicators': df['indicator'].value_counts().to_dict(),
            'districts': df['district'].value_counts().to_dict(),
            'years': df['year'].value_counts().to_dict()
        }
        
        return summary

def main():
    """Main function to run schema normalization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Normalize extracted data')
    parser.add_argument('--input-file', default='data/extracted/all_extracted_data.json',
                       help='Input file with extracted data')
    parser.add_argument('--output-dir', default='data/normalized',
                       help='Output directory for normalized data')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load extracted data
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        return
    
    # Initialize normalizer
    normalizer = SchemaNormalizer(output_dir=args.output_dir)
    
    # Normalize data
    normalized_facts = normalizer.normalize_extracted_data(extracted_data)
    
    # Generate summary
    summary = normalizer.generate_summary_report(normalized_facts)
    
    # Print summary
    print(f"\nNormalization Summary:")
    print(f"Total facts: {summary.get('total_facts', 0)}")
    print(f"Unique indicators: {summary.get('unique_indicators', 0)}")
    print(f"Unique districts: {summary.get('unique_districts', 0)}")
    print(f"Unique years: {summary.get('unique_years', 0)}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
