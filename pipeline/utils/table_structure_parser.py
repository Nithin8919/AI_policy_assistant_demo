#!/usr/bin/env python3
"""
Enhanced Table Structure Parser
Fixes concatenated table text extraction and converts it to proper structured data
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TableStructureParser:
    """
    Parses concatenated table text into structured rows and columns
    Handles common AP education document patterns
    """
    
    def __init__(self):
        # AP district patterns (with variations)
        self.ap_districts = {
            'anantapur', 'ananthpur', 'anantpur',
            'chittoor', 'chitoor', 'chitor',
            'east godavari', 'eastgodavari', 'e.godavari',
            'guntur', 'guntoor',
            'kadapa', 'cudapah', 'cuddapah',
            'krishna', 
            'kurnool', 'kurnul',
            'nellore', 'nelore',
            'prakasam', 'prakasham',
            'srikakulam', 'shrikakulam',
            'visakhapatnam', 'vizag', 'vishakhapatnam',
            'vizianagaram', 'vijayanagaram',
            'west godavari', 'westgodavari', 'w.godavari'
        }
        
        # Common table headers in education documents
        self.common_headers = {
            'sl.no', 'slno', 's.no', 'serial', 'sr.no',
            'district', 'dist', 'district name',
            'application id', 'app id', 'id',
            'applicant name', 'name', 'student name',
            'school', 'school name', 'institution',
            'enrollment', 'enrolment', 'strength',
            'teachers', 'staff', 'faculty',
            'boys', 'girls', 'total',
            'urban', 'rural',
            'primary', 'upper primary', 'secondary',
            'budget', 'allocation', 'amount',
            'year', 'academic year', 'session'
        }
        
        # Numeric patterns
        self.numeric_patterns = [
            r'\d+\.?\d*',  # Regular numbers
            r'\d{1,3}(?:,\d{3})*',  # Numbers with commas
            r'AP\d{12}',  # Application IDs
            r'\d{4}-\d{2}',  # Years like 2023-24
        ]
    
    def parse_concatenated_table(self, raw_text: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Parse concatenated table text into structured format
        
        Args:
            raw_text: Concatenated table text
            context: Additional context (page, document type, etc.)
            
        Returns:
            Structured table data or None if parsing fails
        """
        if not raw_text or len(raw_text.strip()) < 20:
            return None
            
        logger.info(f"🔧 Parsing table text: {len(raw_text)} characters")
        
        # Try different parsing strategies
        strategies = [
            self._parse_numbered_list,
            self._parse_district_table,
            self._parse_application_table,
            self._parse_statistical_table,
            self._parse_budget_table
        ]
        
        for strategy in strategies:
            try:
                result = strategy(raw_text, context)
                if result and self._validate_parsed_table(result):
                    logger.info(f"✅ Successfully parsed with {strategy.__name__}")
                    return result
            except Exception as e:
                logger.warning(f"⚠️ Strategy {strategy.__name__} failed: {e}")
                continue
        
        logger.warning(f"❌ All parsing strategies failed for text: {raw_text[:100]}...")
        return None
    
    def _parse_numbered_list(self, text: str, context: Dict = None) -> Optional[Dict[str, Any]]:
        """Parse tables with numbered rows like 'SL.No District Name ...'"""
        
        # Look for numbered pattern
        if not re.search(r'(?:sl\.?no|s\.?no|serial)', text.lower()):
            return None
            
        logger.info("🔧 Attempting numbered list parsing")
        
        # Extract header line
        lines = text.strip().split('\n')
        potential_header = lines[0] if lines else ""
        
        # Parse header
        headers = self._extract_headers_from_text(potential_header)
        if not headers:
            return None
            
        # Parse rows
        rows = []
        remaining_text = ' '.join(lines[1:]) if len(lines) > 1 else text
        
        # Split by number patterns
        row_pattern = r'(\d+)\s+(.*?)(?=\d+\s+|$)'
        matches = re.findall(row_pattern, remaining_text)
        
        for match in matches:
            sl_no, row_data = match
            row_cells = self._parse_row_data(row_data, len(headers))
            if row_cells:
                rows.append([sl_no] + row_cells)
        
        if not rows:
            return None
            
        return {
            'headers': headers,
            'rows': rows,
            'shape': (len(rows), len(headers)),
            'parsing_method': 'numbered_list',
            'confidence': 0.8
        }
    
    def _parse_district_table(self, text: str, context: Dict = None) -> Optional[Dict[str, Any]]:
        """Parse tables with district-based data"""
        
        # Check if text contains district names
        text_lower = text.lower()
        district_matches = sum(1 for district in self.ap_districts if district in text_lower)
        
        if district_matches < 2:  # Need at least 2 districts
            return None
            
        logger.info(f"🔧 Attempting district table parsing ({district_matches} districts found)")
        
        rows = []
        headers = ['District', 'Value', 'Additional Info']
        
        # Split by district names
        for district in self.ap_districts:
            pattern = rf'\b{re.escape(district)}\b.*?(?=\b(?:' + '|'.join(self.ap_districts) + r')\b|$)'
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            
            for match in matches:
                district_text = match.group()
                
                # Extract numbers from this district's text
                numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', district_text)
                if numbers:
                    # Take first number as primary value
                    value = numbers[0]
                    additional = ', '.join(numbers[1:]) if len(numbers) > 1 else ''
                    rows.append([district.title(), value, additional])
        
        if len(rows) < 2:
            return None
            
        return {
            'headers': headers,
            'rows': rows,
            'shape': (len(rows), len(headers)),
            'parsing_method': 'district_table',
            'confidence': 0.7
        }
    
    def _parse_application_table(self, text: str, context: Dict = None) -> Optional[Dict[str, Any]]:
        """Parse tables with application IDs like 'AP202324000001'"""
        
        # Look for AP application ID pattern
        app_ids = re.findall(r'AP\d{12}', text)
        if len(app_ids) < 2:
            return None
            
        logger.info(f"🔧 Attempting application table parsing ({len(app_ids)} applications found)")
        
        headers = ['Serial', 'District', 'Application ID', 'Applicant Name']
        rows = []
        
        # Pattern to match: Number District AppID Name
        pattern = r'(\d+)\s+([A-Za-z\s]+?)\s+(AP\d{12})\s+([A-Za-z\s]+?)(?=\d+\s+|$)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for match in matches:
            serial, district, app_id, name = match
            rows.append([
                serial.strip(),
                district.strip(),
                app_id.strip(),
                name.strip()
            ])
        
        if not rows:
            return None
            
        return {
            'headers': headers,
            'rows': rows,
            'shape': (len(rows), len(headers)),
            'parsing_method': 'application_table',
            'confidence': 0.9
        }
    
    def _parse_statistical_table(self, text: str, context: Dict = None) -> Optional[Dict[str, Any]]:
        """Parse statistical tables with enrollment, teachers, etc."""
        
        # Look for statistical keywords
        stat_keywords = ['enrollment', 'enrolment', 'teachers', 'schools', 'students', 'boys', 'girls']
        if not any(keyword in text.lower() for keyword in stat_keywords):
            return None
            
        logger.info("🔧 Attempting statistical table parsing")
        
        # Extract structured data
        rows = []
        
        # Pattern for statistical rows: Indicator Value1 Value2 ...
        lines = text.strip().split('\n')
        for line in lines:
            # Look for lines with multiple numbers
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', line)
            if len(numbers) >= 2:
                # Extract text before numbers as indicator
                text_part = re.split(r'\d+', line)[0].strip()
                if text_part:
                    rows.append([text_part] + numbers)
        
        if len(rows) < 2:
            return None
            
        # Create headers based on max row length
        max_cols = max(len(row) for row in rows)
        headers = ['Indicator'] + [f'Value_{i+1}' for i in range(max_cols - 1)]
        
        # Normalize row lengths
        normalized_rows = []
        for row in rows:
            normalized_row = row + [''] * (max_cols - len(row))
            normalized_rows.append(normalized_row[:max_cols])
        
        return {
            'headers': headers,
            'rows': normalized_rows,
            'shape': (len(normalized_rows), len(headers)),
            'parsing_method': 'statistical_table',
            'confidence': 0.6
        }
    
    def _parse_budget_table(self, text: str, context: Dict = None) -> Optional[Dict[str, Any]]:
        """Parse budget/financial tables"""
        
        # Look for financial keywords
        if not any(keyword in text.lower() for keyword in ['budget', 'allocation', 'amount', 'crore', 'lakh', 'rupees']):
            return None
            
        logger.info("🔧 Attempting budget table parsing")
        
        rows = []
        headers = ['Scheme/Department', 'Allocation', 'Currency']
        
        # Pattern for budget lines
        lines = text.strip().split('\n')
        for line in lines:
            # Look for amount patterns
            amount_match = re.search(r'(\d+(?:\.\d+)?)\s*(crore|lakh|rupees?)', line.lower())
            if amount_match:
                amount, currency = amount_match.groups()
                # Text before amount as scheme/department
                scheme = line[:amount_match.start()].strip()
                if scheme:
                    rows.append([scheme, amount, currency])
        
        if len(rows) < 2:
            return None
            
        return {
            'headers': headers,
            'rows': rows,
            'shape': (len(rows), len(headers)),
            'parsing_method': 'budget_table',
            'confidence': 0.7
        }
    
    def _extract_headers_from_text(self, header_text: str) -> List[str]:
        """Extract headers from concatenated header text"""
        
        # Common header patterns
        header_patterns = [
            r'sl\.?\s*no\.?',
            r's\.?\s*no\.?',
            r'district\s*(?:name)?',
            r'application\s*id',
            r'applicant\s*name',
            r'school\s*(?:name)?',
            r'enrollment?',
            r'teachers?',
            r'students?',
            r'amount'
        ]
        
        headers = []
        remaining_text = header_text.lower()
        
        for pattern in header_patterns:
            match = re.search(pattern, remaining_text)
            if match:
                headers.append(match.group().title())
                # Remove matched part
                remaining_text = remaining_text[:match.start()] + remaining_text[match.end():]
        
        # If no pattern matching, split by common separators
        if not headers:
            separators = ['\t', '  ', ' | ', ',']
            for sep in separators:
                if sep in header_text:
                    headers = [h.strip() for h in header_text.split(sep) if h.strip()]
                    break
        
        return headers[:10]  # Limit to reasonable number
    
    def _parse_row_data(self, row_text: str, expected_cols: int) -> List[str]:
        """Parse individual row data"""
        
        # Try different splitting strategies
        strategies = [
            lambda x: re.split(r'\s{2,}', x),  # Multiple spaces
            lambda x: re.split(r'\t', x),      # Tabs
            lambda x: x.split(),               # Single spaces
        ]
        
        for strategy in strategies:
            cells = strategy(row_text.strip())
            if len(cells) >= expected_cols - 1:  # Allow some tolerance
                return cells[:expected_cols - 1]  # Exclude serial number column
        
        # Fallback: extract numbers and text segments
        numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', row_text)
        words = re.findall(r'[A-Za-z][A-Za-z\s]*[A-Za-z]', row_text)
        
        combined = []
        for word in words:
            if word.strip():
                combined.append(word.strip())
        combined.extend(numbers)
        
        return combined[:expected_cols - 1]
    
    def _validate_parsed_table(self, table_data: Dict[str, Any]) -> bool:
        """Validate parsed table structure"""
        
        if not table_data or 'headers' not in table_data or 'rows' not in table_data:
            return False
            
        headers = table_data['headers']
        rows = table_data['rows']
        
        # Check minimum size
        if len(headers) < 2 or len(rows) < 1:
            return False
            
        # Check row consistency
        for row in rows:
            if len(row) != len(headers):
                return False
                
        # Check for some data content
        non_empty_cells = 0
        for row in rows:
            for cell in row:
                if str(cell).strip():
                    non_empty_cells += 1
                    
        return non_empty_cells >= len(rows)  # At least one non-empty cell per row
    
    def enhance_existing_extraction(self, extraction_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance existing extraction data with better table structure
        
        Args:
            extraction_data: List of extracted items from Stage 1
            
        Returns:
            Enhanced data with improved table structures
        """
        logger.info(f"🔧 Enhancing {len(extraction_data)} extracted items")
        
        enhanced_data = []
        enhancement_stats = {
            'total': len(extraction_data),
            'enhanced': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for item in extraction_data:
            try:
                # Skip non-table items
                if item.get('extraction_method') not in ['camelot_stream', 'ocr_tesseract']:
                    enhanced_data.append(item)
                    enhancement_stats['skipped'] += 1
                    continue
                
                # Try to enhance table structure
                enhanced_item = self._enhance_table_item(item)
                if enhanced_item:
                    enhanced_data.append(enhanced_item)
                    enhancement_stats['enhanced'] += 1
                    logger.info(f"✅ Enhanced table: {item.get('table_id', 'unknown')}")
                else:
                    enhanced_data.append(item)  # Keep original if enhancement fails
                    enhancement_stats['failed'] += 1
                    logger.warning(f"⚠️ Enhancement failed for: {item.get('table_id', 'unknown')}")
                    
            except Exception as e:
                logger.error(f"❌ Enhancement error for item: {e}")
                enhanced_data.append(item)  # Keep original
                enhancement_stats['failed'] += 1
        
        logger.info(f"📊 Enhancement complete:")
        logger.info(f"   - Total: {enhancement_stats['total']}")
        logger.info(f"   - Enhanced: {enhancement_stats['enhanced']}")
        logger.info(f"   - Failed: {enhancement_stats['failed']}")
        logger.info(f"   - Skipped: {enhancement_stats['skipped']}")
        
        return enhanced_data
    
    def _enhance_table_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhance individual table item"""
        
        # Check if it's already well-structured
        if self._is_well_structured(item):
            return item
        
        # Try to reconstruct from raw data
        raw_text = self._reconstruct_table_text(item)
        if not raw_text:
            return None
            
        # Parse the reconstructed text
        parsed_table = self.parse_concatenated_table(raw_text, {
            'doc_id': item.get('doc_id'),
            'page': item.get('page'),
            'source_type': item.get('source_type')
        })
        
        if not parsed_table:
            return None
            
        # Merge parsed structure with original metadata
        enhanced_item = item.copy()
        enhanced_item.update({
            'headers': parsed_table['headers'],
            'rows': parsed_table['rows'],
            'shape': parsed_table['shape'],
            'parsing_method': parsed_table.get('parsing_method', 'enhanced'),
            'enhancement_confidence': parsed_table.get('confidence', 0.5),
            'enhanced_at': pd.Timestamp.now().isoformat()
        })
        
        return enhanced_item
    
    def _is_well_structured(self, item: Dict[str, Any]) -> bool:
        """Check if table is already well-structured"""
        
        headers = item.get('headers', [])
        rows = item.get('rows', [])
        
        if not headers or not rows:
            return False
            
        # Check if headers are meaningful (not just column numbers)
        meaningful_headers = sum(1 for h in headers if not str(h).isdigit() and len(str(h)) > 1)
        
        return meaningful_headers >= len(headers) * 0.5
    
    def _reconstruct_table_text(self, item: Dict[str, Any]) -> Optional[str]:
        """Reconstruct table text from rows data"""
        
        try:
            headers = item.get('headers', [])
            rows = item.get('rows', [])
            
            if not rows:
                return None
                
            # Combine headers and rows into text
            text_parts = []
            
            if headers:
                text_parts.append(' '.join(str(h) for h in headers))
                
            for row in rows:
                if row:
                    text_parts.append(' '.join(str(cell) for cell in row if cell))
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Failed to reconstruct table text: {e}")
            return None


def main():
    """Test the table structure parser"""
    
    # Sample concatenated table text
    sample_text = """
    SL.No District Name Application Id Applicant Name 1 ANANTAPUR AP202324000001 JOHN DOE 2 CHITTOOR AP202324000002 JANE SMITH 3 GUNTUR AP202324000003 RAVI KUMAR
    """
    
    parser = TableStructureParser()
    result = parser.parse_concatenated_table(sample_text)
    
    if result:
        print("✅ Parsing successful!")
        print(f"Headers: {result['headers']}")
        print(f"Rows: {result['rows']}")
        print(f"Shape: {result['shape']}")
        print(f"Method: {result['parsing_method']}")
    else:
        print("❌ Parsing failed")


if __name__ == "__main__":
    main()