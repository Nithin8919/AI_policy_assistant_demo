#!/usr/bin/env python3
"""
Advanced Data Quality Validator for AP Education Pipeline
Comprehensive validation rules with anomaly detection and data quality scoring
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import pandas as pd
import numpy as np
from scipy import stats
import re

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    confidence_score: float
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)

@dataclass
class ValidationRule:
    """Defines a validation rule"""
    rule_id: str
    rule_name: str
    rule_type: str  # 'mandatory', 'warning', 'statistical'
    description: str
    severity: str   # 'critical', 'high', 'medium', 'low'

class DataValidator:
    """Comprehensive data quality validator for education statistics"""
    
    def __init__(self):
        # AP district validation data
        self.valid_districts = {
            'Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Kadapa',
            'Krishna', 'Kurnool', 'Nellore', 'Prakasam', 'Srikakulam',
            'Visakhapatnam', 'Vizianagaram', 'West Godavari'
        }
        
        # Valid education indicators with expected ranges
        self.indicator_ranges = {
            'Total Enrollment': {'min': 1, 'max': 500000, 'unit': 'count'},
            'Boys Enrollment': {'min': 1, 'max': 250000, 'unit': 'count'},
            'Girls Enrollment': {'min': 1, 'max': 250000, 'unit': 'count'},
            'Primary Enrollment': {'min': 1, 'max': 300000, 'unit': 'count'},
            'Upper Primary Enrollment': {'min': 1, 'max': 200000, 'unit': 'count'},
            'Secondary Enrollment': {'min': 1, 'max': 150000, 'unit': 'count'},
            'Higher Secondary Enrollment': {'min': 1, 'max': 100000, 'unit': 'count'},
            
            'Gross Enrollment Ratio': {'min': 0, 'max': 150, 'unit': '%'},
            'Net Enrollment Ratio': {'min': 0, 'max': 100, 'unit': '%'},
            'Gender Parity Index': {'min': 0.5, 'max': 1.5, 'unit': 'ratio'},
            'Dropout Rate': {'min': 0, 'max': 50, 'unit': '%'},
            'Retention Rate': {'min': 50, 'max': 100, 'unit': '%'},
            'Transition Rate': {'min': 70, 'max': 100, 'unit': '%'},
            
            'Number of Schools': {'min': 10, 'max': 10000, 'unit': 'count'},
            'Number of Teachers': {'min': 50, 'max': 50000, 'unit': 'count'},
            'Pupil Teacher Ratio': {'min': 10, 'max': 100, 'unit': 'ratio'},
            'Number of Classrooms': {'min': 100, 'max': 100000, 'unit': 'count'},
            
            'Toilet Facilities': {'min': 0, 'max': 100, 'unit': '%'},
            'Drinking Water Facilities': {'min': 0, 'max': 100, 'unit': '%'},
            'Electricity Connection': {'min': 0, 'max': 100, 'unit': '%'},
            'Computer Facilities': {'min': 0, 'max': 100, 'unit': '%'},
            'Library Facilities': {'min': 0, 'max': 100, 'unit': '%'}
        }
        
        # Valid year range for education data
        self.valid_year_range = (2010, datetime.now().year + 1)
        
        # Statistical thresholds for anomaly detection
        self.anomaly_thresholds = {
            'z_score': 3.0,      # Standard deviations for outliers
            'iqr_multiplier': 1.5, # IQR multiplier for outliers
            'cv_threshold': 0.8,    # Coefficient of variation threshold
            'growth_rate': 0.5      # Maximum year-over-year growth rate
        }
        
        # Validation rules
        self.validation_rules = [
            ValidationRule(
                rule_id='DIST_001',
                rule_name='Valid District',
                rule_type='mandatory',
                description='District must be one of the 13 AP districts',
                severity='critical'
            ),
            ValidationRule(
                rule_id='IND_001',
                rule_name='Valid Indicator',
                rule_type='mandatory',
                description='Indicator must be a recognized education metric',
                severity='critical'
            ),
            ValidationRule(
                rule_id='VAL_001',
                rule_name='Value Range',
                rule_type='mandatory',
                description='Value must be within expected range for indicator',
                severity='high'
            ),
            ValidationRule(
                rule_id='YEAR_001',
                rule_name='Valid Year',
                rule_type='mandatory',
                description='Year must be within valid range (2010-present)',
                severity='high'
            ),
            ValidationRule(
                rule_id='UNIT_001',
                rule_name='Consistent Units',
                rule_type='warning',
                description='Unit should match expected unit for indicator',
                severity='medium'
            ),
            ValidationRule(
                rule_id='STAT_001',
                rule_name='Statistical Outlier',
                rule_type='statistical',
                description='Value is statistical outlier compared to other districts',
                severity='low'
            ),
            ValidationRule(
                rule_id='CONS_001',
                rule_name='Logical Consistency',
                rule_type='warning',
                description='Values should be logically consistent (boys + girls = total)',
                severity='medium'
            ),
            ValidationRule(
                rule_id='TEMP_001',
                rule_name='Temporal Consistency',
                rule_type='warning',
                description='Year-over-year changes should be reasonable',
                severity='medium'
            )
        ]
    
    def validate_facts(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate all facts and return comprehensive validation report"""
        logger.info(f"🔍 Validating {len(facts)} facts")
        
        validation_report = {
            'summary': {
                'total_facts': len(facts),
                'valid_facts': 0,
                'invalid_facts': 0,
                'facts_with_warnings': 0,
                'overall_quality_score': 0.0
            },
            'rule_violations': {},
            'quality_metrics': {},
            'anomalies': [],
            'fact_results': [],
            'district_summary': {},
            'indicator_summary': {},
            'year_summary': {}
        }
        
        # Initialize rule violation tracking
        for rule in self.validation_rules:
            validation_report['rule_violations'][rule.rule_id] = {
                'rule_name': rule.rule_name,
                'violations': 0,
                'severity': rule.severity,
                'description': rule.description
            }
        
        # Validate each fact
        individual_results = []
        for i, fact in enumerate(facts):
            result = self._validate_single_fact(fact, facts)
            individual_results.append(result)
            validation_report['fact_results'].append({
                'fact_index': i,
                'fact_id': fact.get('fact_id', f'fact_{i}'),
                'is_valid': result.is_valid,
                'confidence_score': result.confidence_score,
                'errors': result.validation_errors,
                'warnings': result.validation_warnings,
                'anomalies': result.anomalies
            })
            
            # Update rule violations
            for error in result.validation_errors:
                rule_id = error.split(':')[0] if ':' in error else 'UNKNOWN'
                if rule_id in validation_report['rule_violations']:
                    validation_report['rule_violations'][rule_id]['violations'] += 1
            
            # Update summary
            if result.is_valid:
                validation_report['summary']['valid_facts'] += 1
            else:
                validation_report['summary']['invalid_facts'] += 1
                
            if result.validation_warnings:
                validation_report['summary']['facts_with_warnings'] += 1
        
        # Calculate overall quality score
        quality_scores = [r.confidence_score for r in individual_results]
        validation_report['summary']['overall_quality_score'] = np.mean(quality_scores) if quality_scores else 0.0
        
        # Generate quality metrics
        validation_report['quality_metrics'] = self._calculate_quality_metrics(facts, individual_results)
        
        # Generate summaries
        validation_report['district_summary'] = self._generate_district_summary(facts, individual_results)
        validation_report['indicator_summary'] = self._generate_indicator_summary(facts, individual_results)
        validation_report['year_summary'] = self._generate_year_summary(facts, individual_results)
        
        # Detect dataset-level anomalies
        validation_report['anomalies'] = self._detect_dataset_anomalies(facts)
        
        logger.info(f"✅ Validation complete:")
        logger.info(f"   - Valid facts: {validation_report['summary']['valid_facts']}")
        logger.info(f"   - Invalid facts: {validation_report['summary']['invalid_facts']}")
        logger.info(f"   - Quality score: {validation_report['summary']['overall_quality_score']:.3f}")
        
        return validation_report
    
    def _validate_single_fact(self, fact: Dict[str, Any], all_facts: List[Dict[str, Any]]) -> ValidationResult:
        """Validate a single fact"""
        result = ValidationResult(is_valid=True, confidence_score=1.0)
        
        # Extract fact components
        district = fact.get('district', '')
        indicator = fact.get('indicator', '')
        value = fact.get('value')
        year = fact.get('year')
        unit = fact.get('unit', '')
        category = fact.get('category')
        
        # Rule DIST_001: Valid District
        if district not in self.valid_districts:
            result.validation_errors.append(f"DIST_001: Invalid district '{district}'")
            result.is_valid = False
            result.confidence_score -= 0.3
        
        # Rule IND_001: Valid Indicator
        if indicator not in self.indicator_ranges:
            result.validation_errors.append(f"IND_001: Unrecognized indicator '{indicator}'")
            result.is_valid = False
            result.confidence_score -= 0.2
        
        # Rule VAL_001: Value Range
        if value is not None and indicator in self.indicator_ranges:
            range_info = self.indicator_ranges[indicator]
            if not (range_info['min'] <= value <= range_info['max']):
                result.validation_errors.append(
                    f"VAL_001: Value {value} outside valid range [{range_info['min']}, {range_info['max']}] for {indicator}"
                )
                result.is_valid = False
                result.confidence_score -= 0.2
        elif value is None:
            result.validation_errors.append("VAL_001: Missing value")
            result.is_valid = False
            result.confidence_score -= 0.3
        
        # Rule YEAR_001: Valid Year
        if year is None or not (self.valid_year_range[0] <= year <= self.valid_year_range[1]):
            result.validation_errors.append(f"YEAR_001: Invalid year {year}")
            result.is_valid = False
            result.confidence_score -= 0.1
        
        # Rule UNIT_001: Consistent Units
        if indicator in self.indicator_ranges:
            expected_unit = self.indicator_ranges[indicator]['unit']
            if unit != expected_unit:
                result.validation_warnings.append(
                    f"UNIT_001: Unit '{unit}' doesn't match expected '{expected_unit}' for {indicator}"
                )
                result.confidence_score -= 0.05
        
        # Rule STAT_001: Statistical Outlier Detection
        if district in self.valid_districts and indicator in self.indicator_ranges and value is not None:
            is_outlier, outlier_type = self._detect_statistical_outlier(fact, all_facts)
            if is_outlier:
                result.validation_warnings.append(f"STAT_001: Statistical outlier detected ({outlier_type})")
                result.anomalies.append(f"Statistical outlier: {outlier_type}")
                result.confidence_score -= 0.1
        
        # Rule CONS_001: Logical Consistency
        consistency_issues = self._check_logical_consistency(fact, all_facts)
        for issue in consistency_issues:
            result.validation_warnings.append(f"CONS_001: {issue}")
            result.confidence_score -= 0.05
        
        # Rule TEMP_001: Temporal Consistency  
        temporal_issues = self._check_temporal_consistency(fact, all_facts)
        for issue in temporal_issues:
            result.validation_warnings.append(f"TEMP_001: {issue}")
            result.confidence_score -= 0.05
        
        # Ensure confidence score is within bounds
        result.confidence_score = max(0.0, min(1.0, result.confidence_score))
        
        return result
    
    def _detect_statistical_outlier(self, fact: Dict[str, Any], all_facts: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Detect if fact is a statistical outlier"""
        district = fact.get('district')
        indicator = fact.get('indicator')
        value = fact.get('value')
        year = fact.get('year')
        
        if value is None:
            return False, ''
        
        # Get similar facts (same indicator, year)
        similar_facts = [
            f for f in all_facts 
            if f.get('indicator') == indicator 
            and f.get('year') == year 
            and f.get('value') is not None
            and f.get('district') != district  # Exclude self
        ]
        
        if len(similar_facts) < 3:  # Need at least 3 points for statistics
            return False, 'insufficient_data'
        
        values = [f['value'] for f in similar_facts]
        
        # Z-score test
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val > 0:
            z_score = abs(value - mean_val) / std_val
            if z_score > self.anomaly_thresholds['z_score']:
                return True, f'z_score_{z_score:.2f}'
        
        # IQR test
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if iqr > 0:
            lower_bound = q1 - self.anomaly_thresholds['iqr_multiplier'] * iqr
            upper_bound = q3 + self.anomaly_thresholds['iqr_multiplier'] * iqr
            
            if value < lower_bound or value > upper_bound:
                return True, f'iqr_outlier'
        
        return False, ''
    
    def _check_logical_consistency(self, fact: Dict[str, Any], all_facts: List[Dict[str, Any]]) -> List[str]:
        """Check logical consistency between related facts"""
        issues = []
        
        district = fact.get('district')
        indicator = fact.get('indicator')
        value = fact.get('value')
        year = fact.get('year')
        
        if value is None:
            return issues
        
        # Check boys + girls = total enrollment consistency
        if indicator == 'Total Enrollment':
            boys_fact = self._find_matching_fact(all_facts, district, 'Boys Enrollment', year)
            girls_fact = self._find_matching_fact(all_facts, district, 'Girls Enrollment', year)
            
            if boys_fact and girls_fact:
                boys_val = boys_fact.get('value', 0)
                girls_val = girls_fact.get('value', 0)
                total_calculated = boys_val + girls_val
                
                # Allow 5% tolerance
                tolerance = 0.05 * value
                if abs(value - total_calculated) > tolerance:
                    issues.append(
                        f"Total enrollment ({value}) doesn't match boys ({boys_val}) + girls ({girls_val}) = {total_calculated}"
                    )
        
        # Check primary + upper primary <= total enrollment
        if indicator == 'Total Enrollment':
            primary_fact = self._find_matching_fact(all_facts, district, 'Primary Enrollment', year)
            upper_primary_fact = self._find_matching_fact(all_facts, district, 'Upper Primary Enrollment', year)
            
            if primary_fact and upper_primary_fact:
                primary_val = primary_fact.get('value', 0)
                upper_primary_val = upper_primary_fact.get('value', 0)
                elementary_total = primary_val + upper_primary_val
                
                if elementary_total > value * 1.1:  # 10% tolerance
                    issues.append(
                        f"Elementary enrollment ({elementary_total}) exceeds total enrollment ({value})"
                    )
        
        # Check PTR = Students / Teachers consistency
        if indicator == 'Pupil Teacher Ratio':
            enrollment_fact = self._find_matching_fact(all_facts, district, 'Total Enrollment', year)
            teachers_fact = self._find_matching_fact(all_facts, district, 'Number of Teachers', year)
            
            if enrollment_fact and teachers_fact:
                enrollment_val = enrollment_fact.get('value', 0)
                teachers_val = teachers_fact.get('value', 0)
                
                if teachers_val > 0:
                    calculated_ptr = enrollment_val / teachers_val
                    tolerance = 0.1 * value
                    
                    if abs(value - calculated_ptr) > tolerance:
                        issues.append(
                            f"PTR ({value}) doesn't match students ({enrollment_val}) / teachers ({teachers_val}) = {calculated_ptr:.1f}"
                        )
        
        return issues
    
    def _check_temporal_consistency(self, fact: Dict[str, Any], all_facts: List[Dict[str, Any]]) -> List[str]:
        """Check temporal consistency (year-over-year changes)"""
        issues = []
        
        district = fact.get('district')
        indicator = fact.get('indicator')
        value = fact.get('value')
        year = fact.get('year')
        
        if value is None or year is None:
            return issues
        
        # Find previous year's value
        prev_year_fact = self._find_matching_fact(all_facts, district, indicator, year - 1)
        if prev_year_fact:
            prev_value = prev_year_fact.get('value')
            if prev_value and prev_value > 0:
                growth_rate = abs(value - prev_value) / prev_value
                
                if growth_rate > self.anomaly_thresholds['growth_rate']:
                    change_type = 'increase' if value > prev_value else 'decrease'
                    issues.append(
                        f"Large year-over-year {change_type}: {prev_value} ({year-1}) → {value} ({year}) ({growth_rate:.1%} change)"
                    )
        
        return issues
    
    def _find_matching_fact(self, all_facts: List[Dict[str, Any]], district: str, indicator: str, year: int) -> Optional[Dict[str, Any]]:
        """Find fact matching district, indicator, and year"""
        for fact in all_facts:
            if (fact.get('district') == district and 
                fact.get('indicator') == indicator and 
                fact.get('year') == year):
                return fact
        return None
    
    def _calculate_quality_metrics(self, facts: List[Dict[str, Any]], results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate overall data quality metrics"""
        if not facts:
            return {}
        
        total_facts = len(facts)
        valid_facts = sum(1 for r in results if r.is_valid)
        
        # Completeness metrics
        non_null_districts = sum(1 for f in facts if f.get('district'))
        non_null_indicators = sum(1 for f in facts if f.get('indicator'))
        non_null_values = sum(1 for f in facts if f.get('value') is not None)
        non_null_years = sum(1 for f in facts if f.get('year') is not None)
        
        # Consistency metrics
        facts_with_warnings = sum(1 for r in results if r.validation_warnings)
        facts_with_anomalies = sum(1 for r in results if r.anomalies)
        
        # Coverage metrics
        unique_districts = len(set(f.get('district') for f in facts if f.get('district')))
        unique_indicators = len(set(f.get('indicator') for f in facts if f.get('indicator')))
        unique_years = len(set(f.get('year') for f in facts if f.get('year')))
        
        return {
            'validity_rate': valid_facts / total_facts,
            'completeness_district': non_null_districts / total_facts,
            'completeness_indicator': non_null_indicators / total_facts,
            'completeness_value': non_null_values / total_facts,
            'completeness_year': non_null_years / total_facts,
            'warning_rate': facts_with_warnings / total_facts,
            'anomaly_rate': facts_with_anomalies / total_facts,
            'district_coverage': unique_districts / len(self.valid_districts),
            'indicator_coverage': unique_indicators / len(self.indicator_ranges),
            'year_span': unique_years,
            'average_confidence': np.mean([r.confidence_score for r in results])
        }
    
    def _generate_district_summary(self, facts: List[Dict[str, Any]], results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate district-wise validation summary"""
        district_summary = {}
        
        for district in self.valid_districts:
            district_facts = [i for i, f in enumerate(facts) if f.get('district') == district]
            district_results = [results[i] for i in district_facts]
            
            if district_facts:
                district_summary[district] = {
                    'total_facts': len(district_facts),
                    'valid_facts': sum(1 for r in district_results if r.is_valid),
                    'average_confidence': np.mean([r.confidence_score for r in district_results]),
                    'warning_count': sum(len(r.validation_warnings) for r in district_results),
                    'error_count': sum(len(r.validation_errors) for r in district_results),
                    'anomaly_count': sum(len(r.anomalies) for r in district_results)
                }
            else:
                district_summary[district] = {
                    'total_facts': 0,
                    'valid_facts': 0,
                    'average_confidence': 0.0,
                    'warning_count': 0,
                    'error_count': 0,
                    'anomaly_count': 0
                }
        
        return district_summary
    
    def _generate_indicator_summary(self, facts: List[Dict[str, Any]], results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate indicator-wise validation summary"""
        indicator_summary = {}
        
        indicators = set(f.get('indicator') for f in facts if f.get('indicator'))
        
        for indicator in indicators:
            indicator_facts = [i for i, f in enumerate(facts) if f.get('indicator') == indicator]
            indicator_results = [results[i] for i in indicator_facts]
            
            indicator_values = [facts[i].get('value') for i in indicator_facts if facts[i].get('value') is not None]
            
            indicator_summary[indicator] = {
                'total_facts': len(indicator_facts),
                'valid_facts': sum(1 for r in indicator_results if r.is_valid),
                'average_confidence': np.mean([r.confidence_score for r in indicator_results]),
                'value_range': [min(indicator_values), max(indicator_values)] if indicator_values else [None, None],
                'value_mean': np.mean(indicator_values) if indicator_values else None,
                'value_std': np.std(indicator_values) if indicator_values else None,
                'anomaly_count': sum(len(r.anomalies) for r in indicator_results)
            }
        
        return indicator_summary
    
    def _generate_year_summary(self, facts: List[Dict[str, Any]], results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate year-wise validation summary"""
        year_summary = {}
        
        years = set(f.get('year') for f in facts if f.get('year') is not None)
        
        for year in sorted(years):
            year_facts = [i for i, f in enumerate(facts) if f.get('year') == year]
            year_results = [results[i] for i in year_facts]
            
            year_summary[year] = {
                'total_facts': len(year_facts),
                'valid_facts': sum(1 for r in year_results if r.is_valid),
                'average_confidence': np.mean([r.confidence_score for r in year_results]),
                'unique_districts': len(set(facts[i].get('district') for i in year_facts)),
                'unique_indicators': len(set(facts[i].get('indicator') for i in year_facts))
            }
        
        return year_summary
    
    def _detect_dataset_anomalies(self, facts: List[Dict[str, Any]]) -> List[str]:
        """Detect dataset-level anomalies"""
        anomalies = []
        
        if not facts:
            return anomalies
        
        # Check for missing districts
        present_districts = set(f.get('district') for f in facts if f.get('district'))
        missing_districts = self.valid_districts - present_districts
        if missing_districts:
            anomalies.append(f"Missing districts: {', '.join(sorted(missing_districts))}")
        
        # Check for temporal gaps
        years = sorted(set(f.get('year') for f in facts if f.get('year')))
        if len(years) > 1:
            for i in range(1, len(years)):
                gap = years[i] - years[i-1]
                if gap > 1:
                    anomalies.append(f"Temporal gap: {gap} years between {years[i-1]} and {years[i]}")
        
        # Check for indicator imbalance
        indicator_counts = {}
        for fact in facts:
            indicator = fact.get('indicator')
            if indicator:
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
        
        if indicator_counts:
            mean_count = np.mean(list(indicator_counts.values()))
            for indicator, count in indicator_counts.items():
                if count < mean_count * 0.3:  # Less than 30% of average
                    anomalies.append(f"Under-represented indicator: {indicator} ({count} facts)")
        
        return anomalies
    
    def save_validation_report(self, validation_report: Dict[str, Any], output_file: str):
        """Save validation report to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Validation report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save validation report: {e}")


def main():
    """Test the data validator"""
    
    # Sample facts for testing
    sample_facts = [
        {
            'fact_id': 'fact_001',
            'district': 'Anantapur',
            'indicator': 'Total Enrollment',
            'value': 125000,
            'year': 2022,
            'unit': 'count'
        },
        {
            'fact_id': 'fact_002',
            'district': 'Invalid District',  # This will fail validation
            'indicator': 'Total Enrollment',
            'value': 999999,  # This might be an outlier
            'year': 2022,
            'unit': 'count'
        },
        {
            'fact_id': 'fact_003',
            'district': 'Chittoor',
            'indicator': 'Dropout Rate',
            'value': 150,  # This will fail range validation
            'year': 2022,
            'unit': '%'
        }
    ]
    
    validator = DataValidator()
    report = validator.validate_facts(sample_facts)
    
    print(f"✅ Validation complete:")
    print(f"   - Total facts: {report['summary']['total_facts']}")
    print(f"   - Valid facts: {report['summary']['valid_facts']}")
    print(f"   - Quality score: {report['summary']['overall_quality_score']:.3f}")
    
    # Show rule violations
    for rule_id, violation in report['rule_violations'].items():
        if violation['violations'] > 0:
            print(f"   - {rule_id}: {violation['violations']} violations ({violation['severity']})")


if __name__ == "__main__":
    main()