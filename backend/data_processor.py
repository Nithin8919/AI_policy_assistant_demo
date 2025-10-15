#!/usr/bin/env python3
"""
Structured Data Processor for AP Policy Co-Pilot
Handles statistics (UDISE+, NAS, ASER), budget, and audit data with metric normalization
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum
import json

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    UDISE = "udise"
    NAS = "nas"
    ASER = "aser"
    BUDGET = "budget"
    AUDIT = "audit"
    SCERT = "scert"
    CSE = "cse"
    CUSTOM = "custom"

@dataclass
class DataPoint:
    """Single data point with full provenance"""
    data_id: str
    source: DataSourceType
    indicator: str  # Normalized indicator code
    indicator_label: str  # Human-readable name
    value: Any
    unit: Optional[str] = None
    
    # Geographic dimensions
    district: Optional[str] = None
    mandal: Optional[str] = None
    school_code: Optional[str] = None
    
    # Temporal dimensions
    year: Optional[int] = None
    month: Optional[int] = None
    academic_year: Optional[str] = None
    
    # Demographic dimensions
    gender: Optional[str] = None
    category: Optional[str] = None  # SC/ST/OBC/General
    management: Optional[str] = None  # Govt/Private/Aided
    
    # Metadata
    source_document: Optional[str] = None
    page_number: Optional[int] = None
    table_name: Optional[str] = None
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0
    notes: Optional[str] = None

@dataclass
class IndicatorDefinition:
    """Standardized indicator definition"""
    code: str
    label: str
    description: str
    unit: str
    data_type: str  # 'count', 'percentage', 'ratio', 'currency'
    valid_range: Optional[Tuple[float, float]] = None
    category: str = "education"  # 'education', 'infrastructure', 'finance', 'enrollment'
    
class StructuredDataProcessor:
    """
    Processes structured data with:
    1. Indicator normalization
    2. District name fuzzy matching
    3. Temporal alignment
    4. Data validation
    5. Metric standardization
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.indicator_registry = {}
        self.district_mapping = {}
        self.data_points = []
        
        self._initialize_indicators()
        self._initialize_district_mapping()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            "data_dirs": {
                "statistics": "data/statistics",
                "budget": "data/budget_audit",
                "extracted": "data/extracted"
            },
            "validation": {
                "min_confidence": 0.7,
                "allow_missing_districts": False,
                "allow_missing_years": False
            },
            "normalization": {
                "fuzzy_match_threshold": 0.85,
                "auto_fix_typos": True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _initialize_indicators(self):
        """Initialize standard indicator definitions"""
        standard_indicators = [
            # Enrollment indicators
            IndicatorDefinition(
                code="ENROLL_TOTAL",
                label="Total Enrollment",
                description="Total number of students enrolled",
                unit="count",
                data_type="count",
                valid_range=(0, 10000000),
                category="enrollment"
            ),
            IndicatorDefinition(
                code="ENROLL_SC",
                label="SC Enrollment",
                description="Scheduled Caste student enrollment",
                unit="count",
                data_type="count",
                valid_range=(0, 5000000),
                category="enrollment"
            ),
            IndicatorDefinition(
                code="ENROLL_ST",
                label="ST Enrollment",
                description="Scheduled Tribe student enrollment",
                unit="count",
                data_type="count",
                valid_range=(0, 2000000),
                category="enrollment"
            ),
            IndicatorDefinition(
                code="ENROLL_GIRL",
                label="Girl Enrollment",
                description="Female student enrollment",
                unit="count",
                data_type="count",
                valid_range=(0, 5000000),
                category="enrollment"
            ),
            
            # Dropout indicators
            IndicatorDefinition(
                code="DROPOUT_RATE",
                label="Dropout Rate",
                description="Percentage of students who dropped out",
                unit="percentage",
                data_type="percentage",
                valid_range=(0, 100),
                category="education"
            ),
            IndicatorDefinition(
                code="DROPOUT_SC",
                label="SC Dropout Rate",
                description="Dropout rate among SC students",
                unit="percentage",
                data_type="percentage",
                valid_range=(0, 100),
                category="education"
            ),
            IndicatorDefinition(
                code="DROPOUT_ST",
                label="ST Dropout Rate",
                description="Dropout rate among ST students",
                unit="percentage",
                data_type="percentage",
                valid_range=(0, 100),
                category="education"
            ),
            
            # Infrastructure indicators
            IndicatorDefinition(
                code="SCHOOLS_TOTAL",
                label="Total Schools",
                description="Total number of schools",
                unit="count",
                data_type="count",
                valid_range=(0, 100000),
                category="infrastructure"
            ),
            IndicatorDefinition(
                code="SCHOOLS_GOVT",
                label="Government Schools",
                description="Number of government schools",
                unit="count",
                data_type="count",
                valid_range=(0, 80000),
                category="infrastructure"
            ),
            IndicatorDefinition(
                code="SCHOOLS_PRIVATE",
                label="Private Schools",
                description="Number of private schools",
                unit="count",
                data_type="count",
                valid_range=(0, 50000),
                category="infrastructure"
            ),
            
            # Teacher indicators
            IndicatorDefinition(
                code="TEACHERS_TOTAL",
                label="Total Teachers",
                description="Total number of teachers",
                unit="count",
                data_type="count",
                valid_range=(0, 500000),
                category="education"
            ),
            IndicatorDefinition(
                code="PTR",
                label="Pupil-Teacher Ratio",
                description="Average number of students per teacher",
                unit="ratio",
                data_type="ratio",
                valid_range=(1, 100),
                category="education"
            ),
            
            # Budget indicators
            IndicatorDefinition(
                code="BUDGET_ALLOCATION",
                label="Budget Allocation",
                description="Total budget allocated for education",
                unit="INR crores",
                data_type="currency",
                valid_range=(0, 100000),
                category="finance"
            ),
            IndicatorDefinition(
                code="BUDGET_EXPENDITURE",
                label="Budget Expenditure",
                description="Actual expenditure on education",
                unit="INR crores",
                data_type="currency",
                valid_range=(0, 100000),
                category="finance"
            ),
        ]
        
        for indicator in standard_indicators:
            self.indicator_registry[indicator.code] = indicator
        
        logger.info(f"Initialized {len(self.indicator_registry)} standard indicators")
    
    def _initialize_district_mapping(self):
        """Initialize district name variations mapping"""
        ap_districts_canonical = [
            "Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna",
            "Kurnool", "Prakasam", "Nellore", "Srikakulam", "Visakhapatnam",
            "Vizianagaram", "West Godavari", "YSR Kadapa", "Alluri Sitharama Raju"
        ]
        
        # Create mapping with common variations
        for district in ap_districts_canonical:
            self.district_mapping[district.lower()] = district
            self.district_mapping[district.upper()] = district
            self.district_mapping[district.replace(" ", "")] = district
            
            # Add common abbreviations
            if district == "Visakhapatnam":
                self.district_mapping["vizag"] = district
                self.district_mapping["visakha"] = district
            elif district == "YSR Kadapa":
                self.district_mapping["kadapa"] = district
                self.district_mapping["cuddapah"] = district
            elif district == "East Godavari":
                self.district_mapping["eastgodavari"] = district
                self.district_mapping["e.godavari"] = district
            elif district == "West Godavari":
                self.district_mapping["westgodavari"] = district
                self.district_mapping["w.godavari"] = district
        
        logger.info(f"Initialized district mapping with {len(self.district_mapping)} variations")
    
    def normalize_indicator(self, raw_indicator: str) -> Optional[str]:
        """Normalize indicator name to standard code"""
        raw_lower = raw_indicator.lower().strip()
        
        # Direct match
        for code, definition in self.indicator_registry.items():
            if raw_lower == definition.label.lower():
                return code
        
        # Fuzzy matching
        from fuzzywuzzy import fuzz
        
        best_match = None
        best_score = 0
        threshold = self.config["normalization"]["fuzzy_match_threshold"] * 100
        
        for code, definition in self.indicator_registry.items():
            score = fuzz.ratio(raw_lower, definition.label.lower())
            if score > best_score and score >= threshold:
                best_score = score
                best_match = code
        
        if best_match:
            logger.debug(f"Matched '{raw_indicator}' to '{best_match}' (score: {best_score})")
            return best_match
        
        # Pattern-based matching
        if "dropout" in raw_lower:
            if "sc" in raw_lower:
                return "DROPOUT_SC"
            elif "st" in raw_lower:
                return "DROPOUT_ST"
            else:
                return "DROPOUT_RATE"
        
        elif "enroll" in raw_lower:
            if "sc" in raw_lower:
                return "ENROLL_SC"
            elif "st" in raw_lower:
                return "ENROLL_ST"
            elif "girl" in raw_lower or "female" in raw_lower:
                return "ENROLL_GIRL"
            else:
                return "ENROLL_TOTAL"
        
        elif "school" in raw_lower:
            if "govt" in raw_lower or "government" in raw_lower:
                return "SCHOOLS_GOVT"
            elif "private" in raw_lower:
                return "SCHOOLS_PRIVATE"
            else:
                return "SCHOOLS_TOTAL"
        
        elif "teacher" in raw_lower:
            if "ratio" in raw_lower or "ptr" in raw_lower:
                return "PTR"
            else:
                return "TEACHERS_TOTAL"
        
        elif "budget" in raw_lower:
            if "expend" in raw_lower:
                return "BUDGET_EXPENDITURE"
            else:
                return "BUDGET_ALLOCATION"
        
        logger.warning(f"Could not normalize indicator: '{raw_indicator}'")
        return None
    
    def normalize_district(self, raw_district: str) -> Optional[str]:
        """Normalize district name to canonical form"""
        if not raw_district:
            return None
        
        raw_clean = raw_district.strip()
        
        # Direct lookup
        if raw_clean.lower() in self.district_mapping:
            return self.district_mapping[raw_clean.lower()]
        
        # Fuzzy matching
        from fuzzywuzzy import fuzz
        
        best_match = None
        best_score = 0
        threshold = self.config["normalization"]["fuzzy_match_threshold"] * 100
        
        canonical_districts = set(self.district_mapping.values())
        for canonical in canonical_districts:
            score = fuzz.ratio(raw_clean.lower(), canonical.lower())
            if score > best_score and score >= threshold:
                best_score = score
                best_match = canonical
        
        if best_match:
            logger.debug(f"Matched district '{raw_district}' to '{best_match}' (score: {best_score})")
            return best_match
        
        logger.warning(f"Could not normalize district: '{raw_district}'")
        return None
    
    def extract_year(self, text: str) -> Optional[int]:
        """Extract year from various formats"""
        # Try YYYY format
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            return int(year_match.group(0))
        
        # Try YYYY-YY format (e.g., 2016-17)
        ay_match = re.search(r'\b(19|20)(\d{2})-(\d{2})\b', text)
        if ay_match:
            return int(ay_match.group(1) + ay_match.group(2))
        
        return None
    
    def process_csv(
        self,
        csv_path: str,
        source_type: DataSourceType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DataPoint]:
        """Process CSV file with structured data"""
        logger.info(f"Processing CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        data_points = []
        
        # Detect column structure
        columns = df.columns.tolist()
        
        # Try to identify key columns
        district_col = self._find_column(columns, ['district', 'dist', 'region'])
        indicator_col = self._find_column(columns, ['indicator', 'metric', 'parameter', 'variable'])
        value_col = self._find_column(columns, ['value', 'count', 'number', 'total'])
        year_col = self._find_column(columns, ['year', 'yr', 'academic_year'])
        
        if not indicator_col or not value_col:
            logger.error(f"Could not identify indicator or value columns in {csv_path}")
            return data_points
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Extract indicator
                raw_indicator = str(row[indicator_col])
                indicator_code = self.normalize_indicator(raw_indicator)
                
                if not indicator_code:
                    logger.warning(f"Skipping row {idx}: unknown indicator '{raw_indicator}'")
                    continue
                
                # Extract value
                value = row[value_col]
                if pd.isna(value):
                    continue
                
                # Extract district
                district = None
                if district_col:
                    raw_district = str(row[district_col])
                    district = self.normalize_district(raw_district)
                
                # Extract year
                year = None
                if year_col:
                    year_value = row[year_col]
                    if isinstance(year_value, (int, float)):
                        year = int(year_value)
                    else:
                        year = self.extract_year(str(year_value))
                
                # Create data point
                data_point = DataPoint(
                    data_id=self._generate_data_id(),
                    source=source_type,
                    indicator=indicator_code,
                    indicator_label=self.indicator_registry[indicator_code].label,
                    value=value,
                    unit=self.indicator_registry[indicator_code].unit,
                    district=district,
                    year=year,
                    source_document=csv_path,
                    confidence=0.9 if district and year else 0.7
                )
                
                # Add metadata if provided
                if metadata:
                    data_point.notes = json.dumps(metadata)
                
                # Validate
                if self._validate_data_point(data_point):
                    data_points.append(data_point)
                    self.data_points.append(data_point)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        logger.info(f"Extracted {len(data_points)} data points from {csv_path}")
        return data_points
    
    def process_json_facts(
        self,
        json_path: str,
        source_type: DataSourceType
    ) -> List[DataPoint]:
        """Process JSON file with extracted facts"""
        logger.info(f"Processing JSON facts: {json_path}")
        
        with open(json_path, 'r') as f:
            facts = json.load(f)
        
        data_points = []
        
        for fact in facts:
            try:
                # Extract components
                raw_indicator = fact.get('indicator', '')
                indicator_code = self.normalize_indicator(raw_indicator)
                
                if not indicator_code:
                    continue
                
                # Create data point
                data_point = DataPoint(
                    data_id=fact.get('id', self._generate_data_id()),
                    source=source_type,
                    indicator=indicator_code,
                    indicator_label=self.indicator_registry[indicator_code].label,
                    value=fact.get('value'),
                    unit=self.indicator_registry[indicator_code].unit,
                    district=self.normalize_district(fact.get('district', '')),
                    year=self.extract_year(str(fact.get('year', ''))),
                    source_document=fact.get('source', json_path),
                    page_number=fact.get('page'),
                    confidence=fact.get('confidence', 0.8)
                )
                
                if self._validate_data_point(data_point):
                    data_points.append(data_point)
                    self.data_points.append(data_point)
                
            except Exception as e:
                logger.error(f"Error processing fact: {e}")
                continue
        
        logger.info(f"Extracted {len(data_points)} data points from {json_path}")
        return data_points
    
    def _find_column(self, columns: List[str], keywords: List[str]) -> Optional[str]:
        """Find column matching any of the keywords"""
        for col in columns:
            col_lower = col.lower()
            for keyword in keywords:
                if keyword in col_lower:
                    return col
        return None
    
    def _validate_data_point(self, data_point: DataPoint) -> bool:
        """Validate data point against indicator definition"""
        indicator_def = self.indicator_registry.get(data_point.indicator)
        
        if not indicator_def:
            return False
        
        # Check required fields
        if not self.config["validation"]["allow_missing_districts"] and not data_point.district:
            logger.warning(f"Data point missing district: {data_point.data_id}")
            return False
        
        if not self.config["validation"]["allow_missing_years"] and not data_point.year:
            logger.warning(f"Data point missing year: {data_point.data_id}")
            return False
        
        # Validate value range
        if indicator_def.valid_range:
            try:
                value_numeric = float(data_point.value)
                min_val, max_val = indicator_def.valid_range
                if not (min_val <= value_numeric <= max_val):
                    logger.warning(
                        f"Value {value_numeric} outside valid range "
                        f"{indicator_def.valid_range} for {indicator_def.code}"
                    )
                    data_point.confidence *= 0.5  # Reduce confidence but don't reject
            except (ValueError, TypeError):
                pass
        
        # Confidence threshold
        if data_point.confidence < self.config["validation"]["min_confidence"]:
            return False
        
        return True
    
    def _generate_data_id(self) -> str:
        """Generate unique data ID"""
        import uuid
        return str(uuid.uuid4())
    
    def get_data_by_indicator(
        self,
        indicator: str,
        district: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[DataPoint]:
        """Query data points by indicator"""
        results = [dp for dp in self.data_points if dp.indicator == indicator]
        
        if district:
            results = [dp for dp in results if dp.district == district]
        
        if year:
            results = [dp for dp in results if dp.year == year]
        
        return results
    
    def get_time_series(
        self,
        indicator: str,
        district: Optional[str] = None
    ) -> pd.DataFrame:
        """Get time series for an indicator"""
        data_points = self.get_data_by_indicator(indicator, district=district)
        
        if not data_points:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            'year': dp.year,
            'value': dp.value,
            'district': dp.district,
            'source': dp.source.value
        } for dp in data_points if dp.year])
        
        return df.sort_values('year')
    
    def get_district_comparison(
        self,
        indicator: str,
        year: int
    ) -> pd.DataFrame:
        """Compare indicator across districts for a given year"""
        data_points = self.get_data_by_indicator(indicator, year=year)
        
        if not data_points:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            'district': dp.district,
            'value': dp.value,
            'source': dp.source.value
        } for dp in data_points if dp.district])
        
        return df.sort_values('district')
    
    def export_to_weaviate_format(self, data_point: DataPoint) -> Dict[str, Any]:
        """Export data point to Weaviate format"""
        return {
            "data_id": data_point.data_id,
            "source": data_point.source.value,
            "indicator": data_point.indicator,
            "indicator_label": data_point.indicator_label,
            "value": str(data_point.value),
            "unit": data_point.unit,
            "district": data_point.district,
            "year": data_point.year,
            "gender": data_point.gender,
            "category": data_point.category,
            "management": data_point.management,
            "source_document": data_point.source_document,
            "page_number": data_point.page_number,
            "confidence": data_point.confidence,
            "text": f"{data_point.indicator_label}: {data_point.value} {data_point.unit or ''} "
                    f"({data_point.district or 'All districts'}, {data_point.year or 'Year N/A'})"
        }
    
    def verify_corpus(self) -> bool:
        """Verify data corpus is loaded"""
        return len(self.data_points) > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed data"""
        return {
            "total_data_points": len(self.data_points),
            "unique_indicators": len(set(dp.indicator for dp in self.data_points)),
            "unique_districts": len(set(dp.district for dp in self.data_points if dp.district)),
            "year_range": (
                min((dp.year for dp in self.data_points if dp.year), default=None),
                max((dp.year for dp in self.data_points if dp.year), default=None)
            ),
            "sources": {
                source: len([dp for dp in self.data_points if dp.source.value == source])
                for source in set(dp.source.value for dp in self.data_points)
            },
            "avg_confidence": np.mean([dp.confidence for dp in self.data_points])
        }

if __name__ == "__main__":
    # Test processing
    processor = StructuredDataProcessor()
    
    # Example: Process UDISE+ data
    data_points = processor.process_csv(
        "data/statistics/udise_2022_23.csv",
        DataSourceType.UDISE,
        metadata={"academic_year": "2022-23"}
    )
    
    print(f"\nProcessed {len(data_points)} data points")
    print(f"\nStatistics: {json.dumps(processor.get_statistics(), indent=2)}")
    
    # Example queries
    print("\n=== Dropout rates for SC students ===")
    sc_dropout = processor.get_data_by_indicator("DROPOUT_SC")
    for dp in sc_dropout[:5]:
        print(f"{dp.district} ({dp.year}): {dp.value}%")