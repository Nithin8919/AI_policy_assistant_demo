#!/usr/bin/env python3
"""
Enhanced Data Processing Pipeline for AP Policy Co-Pilot
Integrates all advanced processing components: table parsing, legal processing, normalization, and validation
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse

# Add pipeline utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

# Import enhanced components
from table_structure_parser import TableStructureParser
from enhanced_legal_processor import EnhancedLegalProcessor
from data_normalizer import DataNormalizer
from data_validator import DataValidator

logger = logging.getLogger(__name__)

class EnhancedPipeline:
    """
    Enhanced data processing pipeline with advanced features:
    1. Improved table structure parsing
    2. Legal document analysis with GO supersession tracking
    3. District-year-metric normalization
    4. Comprehensive data validation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize processors
        self.table_parser = TableStructureParser()
        self.legal_processor = EnhancedLegalProcessor()
        self.normalizer = DataNormalizer()
        self.validator = DataValidator()
        
        # Setup directories
        self.data_dirs = {
            'extracted': Path('data/extracted'),
            'enhanced': Path('data/enhanced'),
            'normalized': Path('data/normalized'), 
            'validated': Path('data/validated'),
            'reports': Path('reports')
        }
        
        # Create directories
        for dir_path in self.data_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Pipeline statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'stages_completed': [],
            'total_facts_input': 0,
            'total_facts_output': 0,
            'enhancement_rate': 0.0,
            'validation_pass_rate': 0.0,
            'processing_time': 0.0
        }
    
    def run_enhanced_pipeline(self, input_data_file: str = None) -> Dict[str, Any]:
        """
        Run the complete enhanced pipeline
        
        Args:
            input_data_file: Path to extracted data file (default: data/extracted/all_extracted_data.json)
            
        Returns:
            Pipeline execution report
        """
        logger.info("🚀 Starting Enhanced Data Processing Pipeline")
        self.stats['start_time'] = datetime.now()
        
        try:
            # Stage 1: Load and enhance table structures
            logger.info("📊 STAGE 1: Enhanced Table Structure Parsing")
            enhanced_data = self._stage_1_enhance_tables(input_data_file)
            self.stats['stages_completed'].append('table_enhancement')
            
            # Stage 2: Legal document analysis
            logger.info("⚖️ STAGE 2: Legal Document Analysis")
            legal_analysis = self._stage_2_legal_analysis(enhanced_data)
            self.stats['stages_completed'].append('legal_analysis')
            
            # Stage 3: Data normalization
            logger.info("🔄 STAGE 3: Data Normalization")
            normalized_facts = self._stage_3_normalization(enhanced_data)
            self.stats['stages_completed'].append('normalization')
            
            # Stage 4: Data validation
            logger.info("✅ STAGE 4: Data Validation")
            validation_report = self._stage_4_validation(normalized_facts)
            self.stats['stages_completed'].append('validation')
            
            # Stage 5: Generate comprehensive report
            logger.info("📋 STAGE 5: Report Generation")
            pipeline_report = self._stage_5_reporting(enhanced_data, legal_analysis, normalized_facts, validation_report)
            self.stats['stages_completed'].append('reporting')
            
            self.stats['end_time'] = datetime.now()
            self.stats['processing_time'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            logger.info("🎉 Enhanced Pipeline Completed Successfully!")
            self._log_pipeline_summary()
            
            return pipeline_report
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def _stage_1_enhance_tables(self, input_data_file: str = None) -> List[Dict[str, Any]]:
        """Stage 1: Enhance table structures"""
        
        # Load extracted data
        if input_data_file is None:
            input_data_file = self.data_dirs['extracted'] / 'all_extracted_data.json'
        
        if not Path(input_data_file).exists():
            raise FileNotFoundError(f"Input data file not found: {input_data_file}")
        
        logger.info(f"📂 Loading data from: {input_data_file}")
        with open(input_data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Flatten data structure if needed
        all_items = []
        if isinstance(raw_data, dict):
            for doc_name, items in raw_data.items():
                if isinstance(items, list):
                    for item in items:
                        item['source_document'] = doc_name
                        all_items.append(item)
                else:
                    all_items.append(items)
        else:
            all_items = raw_data
        
        logger.info(f"📊 Processing {len(all_items)} extracted items")
        self.stats['total_facts_input'] = len(all_items)
        
        # Enhance table structures
        enhanced_items = self.table_parser.enhance_existing_extraction(all_items)
        
        # Calculate enhancement statistics
        enhanced_count = sum(1 for item in enhanced_items if item.get('enhanced_at'))
        self.stats['enhancement_rate'] = enhanced_count / len(all_items) if all_items else 0
        
        # Save enhanced data
        output_file = self.data_dirs['enhanced'] / 'enhanced_extracted_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_items, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Stage 1 complete: {enhanced_count}/{len(all_items)} items enhanced")
        logger.info(f"💾 Enhanced data saved to: {output_file}")
        
        return enhanced_items
    
    def _stage_2_legal_analysis(self, enhanced_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 2: Legal document analysis"""
        
        legal_analysis = {
            'documents_analyzed': 0,
            'go_supersessions': [],
            'legal_references': [],
            'legal_hierarchies': [],
            'definitions': {},
            'procedures': [],
            'document_analyses': {}
        }
        
        # Group data by document
        documents = {}
        for item in enhanced_data:
            doc_id = item.get('doc_id') or item.get('source_document', 'unknown')
            if doc_id not in documents:
                documents[doc_id] = []
            documents[doc_id].append(item)
        
        logger.info(f"📑 Analyzing {len(documents)} documents for legal content")
        
        for doc_id, items in documents.items():
            try:
                # Combine text from all items in document
                doc_text = self._combine_document_text(items)
                
                if len(doc_text.strip()) < 100:  # Skip very short documents
                    continue
                
                # Perform legal analysis
                analysis = self.legal_processor.analyze_legal_document(doc_text, doc_id)
                legal_analysis['document_analyses'][doc_id] = analysis
                
                # Aggregate results
                legal_analysis['go_supersessions'].extend(analysis.get('go_supersessions', []))
                legal_analysis['legal_references'].extend(analysis.get('legal_references', []))
                legal_analysis['legal_hierarchies'].extend(analysis.get('legal_hierarchy', []))
                legal_analysis['definitions'].update(analysis.get('definitions', {}))
                legal_analysis['procedures'].extend(analysis.get('procedures', []))
                
                legal_analysis['documents_analyzed'] += 1
                
                logger.info(f"📄 Analyzed {doc_id}: {analysis.get('document_type', 'unknown')} document")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to analyze document {doc_id}: {e}")
                continue
        
        # Save legal analysis
        output_file = self.data_dirs['enhanced'] / 'legal_analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            # Convert dataclasses to dicts for JSON serialization
            serializable_analysis = self._make_json_serializable(legal_analysis)
            json.dump(serializable_analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Stage 2 complete: {legal_analysis['documents_analyzed']} documents analyzed")
        logger.info(f"   - GO supersessions: {len(legal_analysis['go_supersessions'])}")
        logger.info(f"   - Legal references: {len(legal_analysis['legal_references'])}")
        logger.info(f"   - Definitions: {len(legal_analysis['definitions'])}")
        logger.info(f"💾 Legal analysis saved to: {output_file}")
        
        return legal_analysis
    
    def _stage_3_normalization(self, enhanced_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 3: Data normalization"""
        
        logger.info(f"🔄 Normalizing {len(enhanced_data)} enhanced items")
        
        # Normalize data into standard facts
        normalized_facts = self.normalizer.normalize_extracted_data(enhanced_data)
        
        # Convert NormalizedFact objects to dictionaries
        facts_data = []
        for fact in normalized_facts:
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
        
        # Save normalized facts
        output_file = self.data_dirs['normalized'] / 'normalized_facts.json'
        self.normalizer.save_normalized_facts(normalized_facts, str(output_file))
        
        self.stats['total_facts_output'] = len(facts_data)
        
        logger.info(f"✅ Stage 3 complete: {len(facts_data)} normalized facts created")
        logger.info(f"💾 Normalized facts saved to: {output_file}")
        
        return facts_data
    
    def _stage_4_validation(self, normalized_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 4: Data validation"""
        
        logger.info(f"✅ Validating {len(normalized_facts)} normalized facts")
        
        # Validate facts
        validation_report = self.validator.validate_facts(normalized_facts)
        
        # Calculate validation statistics
        self.stats['validation_pass_rate'] = validation_report['summary']['overall_quality_score']
        
        # Save validation report
        output_file = self.data_dirs['validated'] / 'validation_report.json'
        self.validator.save_validation_report(validation_report, str(output_file))
        
        # Save validated facts (only those that passed validation)
        valid_facts = []
        for i, fact in enumerate(normalized_facts):
            fact_result = validation_report['fact_results'][i]
            if fact_result['is_valid']:
                valid_facts.append(fact)
        
        validated_facts_file = self.data_dirs['validated'] / 'validated_facts.json'
        with open(validated_facts_file, 'w', encoding='utf-8') as f:
            json.dump(valid_facts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Stage 4 complete: {len(valid_facts)}/{len(normalized_facts)} facts passed validation")
        logger.info(f"   - Overall quality score: {validation_report['summary']['overall_quality_score']:.3f}")
        logger.info(f"💾 Validation report saved to: {output_file}")
        logger.info(f"💾 Validated facts saved to: {validated_facts_file}")
        
        return validation_report
    
    def _stage_5_reporting(self, enhanced_data: List[Dict[str, Any]], legal_analysis: Dict[str, Any], 
                          normalized_facts: List[Dict[str, Any]], validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 5: Generate comprehensive pipeline report"""
        
        pipeline_report = {
            'pipeline_metadata': {
                'execution_time': self.stats['processing_time'],
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': self.stats['end_time'].isoformat(),
                'stages_completed': self.stats['stages_completed'],
                'pipeline_version': '2.0_enhanced'
            },
            'data_flow_summary': {
                'input_items': self.stats['total_facts_input'],
                'enhanced_items': len(enhanced_data),
                'normalized_facts': len(normalized_facts),
                'validated_facts': validation_report['summary']['valid_facts'],
                'enhancement_rate': self.stats['enhancement_rate'],
                'validation_pass_rate': self.stats['validation_pass_rate']
            },
            'table_enhancement_summary': self._summarize_table_enhancements(enhanced_data),
            'legal_analysis_summary': self._summarize_legal_analysis(legal_analysis),
            'normalization_summary': self._summarize_normalization(normalized_facts),
            'validation_summary': validation_report['summary'],
            'quality_metrics': validation_report['quality_metrics'],
            'data_coverage': {
                'districts': validation_report['district_summary'],
                'indicators': validation_report['indicator_summary'],
                'years': validation_report['year_summary']
            },
            'recommendations': self._generate_recommendations(enhanced_data, legal_analysis, normalized_facts, validation_report)
        }
        
        # Save pipeline report
        output_file = self.data_dirs['reports'] / f'pipeline_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_report, f, indent=2, ensure_ascii=False, default=str)
        
        # Generate human-readable summary
        self._generate_summary_report(pipeline_report)
        
        logger.info(f"✅ Stage 5 complete: Comprehensive report generated")
        logger.info(f"💾 Pipeline report saved to: {output_file}")
        
        return pipeline_report
    
    def _combine_document_text(self, items: List[Dict[str, Any]]) -> str:
        """Combine text from document items"""
        text_parts = []
        
        for item in items:
            # Get text from different sources
            text = item.get('text', '') or item.get('content', '')
            
            # For table items, combine headers and rows
            if not text and 'headers' in item and 'rows' in item:
                headers = item.get('headers', [])
                rows = item.get('rows', [])
                
                if headers:
                    text_parts.append(' '.join(str(h) for h in headers))
                
                for row in rows:
                    if row:
                        text_parts.append(' '.join(str(cell) for cell in row))
            elif text and len(text.strip()) > 10:
                text_parts.append(text.strip())
        
        return '\n\n'.join(text_parts)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, '__dict__'):
            return {k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (datetime, type(None))):
            return str(obj) if obj else None
        else:
            return obj
    
    def _summarize_table_enhancements(self, enhanced_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize table enhancement results"""
        
        total_items = len(enhanced_data)
        enhanced_items = sum(1 for item in enhanced_data if item.get('enhanced_at'))
        table_items = sum(1 for item in enhanced_data if item.get('extraction_method') in ['camelot_stream', 'ocr_tesseract'])
        
        enhancement_methods = {}
        for item in enhanced_data:
            if item.get('parsing_method'):
                method = item['parsing_method']
                enhancement_methods[method] = enhancement_methods.get(method, 0) + 1
        
        return {
            'total_items': total_items,
            'table_items': table_items,
            'enhanced_items': enhanced_items,
            'enhancement_rate': enhanced_items / total_items if total_items > 0 else 0,
            'enhancement_methods': enhancement_methods
        }
    
    def _summarize_legal_analysis(self, legal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize legal analysis results"""
        
        document_types = {}
        for doc_id, analysis in legal_analysis.get('document_analyses', {}).items():
            doc_type = analysis.get('document_type', 'unknown')
            document_types[doc_type] = document_types.get(doc_type, 0) + 1
        
        return {
            'documents_analyzed': legal_analysis.get('documents_analyzed', 0),
            'document_types': document_types,
            'go_supersessions_found': len(legal_analysis.get('go_supersessions', [])),
            'legal_references_found': len(legal_analysis.get('legal_references', [])),
            'definitions_extracted': len(legal_analysis.get('definitions', {})),
            'procedures_found': len(legal_analysis.get('procedures', []))
        }
    
    def _summarize_normalization(self, normalized_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize normalization results"""
        
        districts = set(fact.get('district') for fact in normalized_facts if fact.get('district'))
        indicators = set(fact.get('indicator') for fact in normalized_facts if fact.get('indicator'))
        years = set(fact.get('year') for fact in normalized_facts if fact.get('year'))
        
        return {
            'total_facts': len(normalized_facts),
            'unique_districts': len(districts),
            'unique_indicators': len(indicators),
            'year_range': [min(years), max(years)] if years else [None, None],
            'districts_covered': sorted(districts),
            'indicators_covered': sorted(indicators)
        }
    
    def _generate_recommendations(self, enhanced_data: List[Dict[str, Any]], legal_analysis: Dict[str, Any],
                                normalized_facts: List[Dict[str, Any]], validation_report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on pipeline results"""
        
        recommendations = []
        
        # Data quality recommendations
        quality_score = validation_report['summary']['overall_quality_score']
        if quality_score < 0.7:
            recommendations.append(f"Data quality score ({quality_score:.2f}) is below threshold. Review extraction methods and source document quality.")
        
        # Coverage recommendations
        district_coverage = validation_report['quality_metrics']['district_coverage']
        if district_coverage < 0.8:
            recommendations.append(f"District coverage ({district_coverage:.1%}) is incomplete. Ensure all 13 AP districts are represented in source documents.")
        
        # Enhancement recommendations
        enhancement_rate = self.stats['enhancement_rate']
        if enhancement_rate < 0.5:
            recommendations.append(f"Table enhancement rate ({enhancement_rate:.1%}) is low. Consider improving table extraction methods.")
        
        # Legal analysis recommendations
        go_supersessions = len(legal_analysis.get('go_supersessions', []))
        if go_supersessions > 0:
            recommendations.append(f"Found {go_supersessions} GO supersessions. Update knowledge base to reflect current policy hierarchy.")
        
        # Temporal recommendations
        years = set(fact.get('year') for fact in normalized_facts if fact.get('year'))
        if years and max(years) < datetime.now().year - 1:
            recommendations.append(f"Most recent data is from {max(years)}. Consider adding more recent documents to improve currency.")
        
        # Anomaly recommendations
        anomaly_count = sum(len(result.get('anomalies', [])) for result in validation_report.get('fact_results', []))
        if anomaly_count > len(normalized_facts) * 0.1:
            recommendations.append(f"High number of anomalies detected ({anomaly_count}). Review data sources for consistency.")
        
        return recommendations
    
    def _generate_summary_report(self, pipeline_report: Dict[str, Any]):
        """Generate human-readable summary report"""
        
        summary_lines = [
            "=" * 80,
            "AP POLICY CO-PILOT: ENHANCED PIPELINE EXECUTION REPORT",
            "=" * 80,
            "",
            f"Execution Time: {pipeline_report['pipeline_metadata']['execution_time']:.2f} seconds",
            f"Pipeline Version: {pipeline_report['pipeline_metadata']['pipeline_version']}",
            f"Stages Completed: {', '.join(pipeline_report['pipeline_metadata']['stages_completed'])}",
            "",
            "DATA FLOW SUMMARY:",
            f"  • Input Items: {pipeline_report['data_flow_summary']['input_items']:,}",
            f"  • Enhanced Items: {pipeline_report['data_flow_summary']['enhanced_items']:,}",
            f"  • Normalized Facts: {pipeline_report['data_flow_summary']['normalized_facts']:,}",
            f"  • Validated Facts: {pipeline_report['data_flow_summary']['validated_facts']:,}",
            f"  • Enhancement Rate: {pipeline_report['data_flow_summary']['enhancement_rate']:.1%}",
            f"  • Validation Pass Rate: {pipeline_report['data_flow_summary']['validation_pass_rate']:.1%}",
            "",
            "QUALITY METRICS:",
            f"  • Completeness: {pipeline_report['quality_metrics']['completeness_value']:.1%}",
            f"  • District Coverage: {pipeline_report['quality_metrics']['district_coverage']:.1%}",
            f"  • Indicator Coverage: {pipeline_report['quality_metrics']['indicator_coverage']:.1%}",
            f"  • Average Confidence: {pipeline_report['quality_metrics']['average_confidence']:.3f}",
            "",
            "KEY FINDINGS:",
            f"  • Documents Analyzed: {pipeline_report['legal_analysis_summary']['documents_analyzed']}",
            f"  • GO Supersessions: {pipeline_report['legal_analysis_summary']['go_supersessions_found']}",
            f"  • Legal References: {pipeline_report['legal_analysis_summary']['legal_references_found']}",
            f"  • Districts Covered: {pipeline_report['normalization_summary']['unique_districts']}/13",
            f"  • Indicators Covered: {pipeline_report['normalization_summary']['unique_indicators']}",
            "",
            "RECOMMENDATIONS:",
        ]
        
        for i, rec in enumerate(pipeline_report['recommendations'], 1):
            summary_lines.append(f"  {i}. {rec}")
        
        summary_lines.extend([
            "",
            "=" * 80
        ])
        
        # Save summary report
        summary_file = self.data_dirs['reports'] / f'pipeline_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        # Also print to console
        print('\n'.join(summary_lines))
        
        logger.info(f"📋 Human-readable summary saved to: {summary_file}")
    
    def _log_pipeline_summary(self):
        """Log pipeline execution summary"""
        
        logger.info("🎯 PIPELINE EXECUTION SUMMARY:")
        logger.info(f"   ⏱️ Total processing time: {self.stats['processing_time']:.2f} seconds")
        logger.info(f"   📊 Input items: {self.stats['total_facts_input']:,}")
        logger.info(f"   📈 Output facts: {self.stats['total_facts_output']:,}")
        logger.info(f"   🔧 Enhancement rate: {self.stats['enhancement_rate']:.1%}")
        logger.info(f"   ✅ Validation pass rate: {self.stats['validation_pass_rate']:.1%}")
        logger.info(f"   🏁 Stages completed: {', '.join(self.stats['stages_completed'])}")


def main():
    """Main function to run enhanced pipeline"""
    
    parser = argparse.ArgumentParser(description='Enhanced Data Processing Pipeline for AP Policy Co-Pilot')
    parser.add_argument('--input-file', help='Input extracted data file')
    parser.add_argument('--config-file', help='Pipeline configuration file')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_pipeline.log')
        ]
    )
    
    # Load configuration if provided
    config = {}
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    
    try:
        # Initialize and run pipeline
        logger.info("🚀 Initializing Enhanced Data Processing Pipeline")
        pipeline = EnhancedPipeline(config)
        
        # Run pipeline
        report = pipeline.run_enhanced_pipeline(args.input_file)
        
        logger.info("🎉 Enhanced Pipeline completed successfully!")
        logger.info(f"📋 Final report available in: {pipeline.data_dirs['reports']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())