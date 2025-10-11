#!/usr/bin/env python3
"""
Production Pipeline Orchestrator
Runs the complete AP education policy intelligence pipeline
"""
import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import argparse

# Add pipeline stages to path
sys.path.append(str(Path(__file__).parent / "stages"))

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Production-ready pipeline orchestrator for AP education policy intelligence"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline stages
        self.stages = [
            {
                "name": "extract_tables",
                "script": "1_extract_tables.py",
                "description": "Extract tables and text from PDFs",
                "input": "data/preprocessed/documents",
                "output": "data/extracted"
            },
            {
                "name": "normalize_schema",
                "script": "2_normalize_schema.py",
                "description": "Normalize extracted data into unified schema",
                "input": "data/extracted/all_extracted_data.json",
                "output": "data/normalized"
            },
            {
                "name": "build_fact_table",
                "script": "3_build_fact_table.py",
                "description": "Build PostgreSQL bridge table with pgvector",
                "input": "data/normalized/normalized_facts.json",
                "output": "data/bridge_table"
            },
            {
                "name": "load_neo4j",
                "script": "4_load_neo4j.py",
                "description": "Load facts into Neo4j knowledge graph",
                "input": "data/normalized/normalized_facts.json",
                "output": "data/neo4j"
            },
            {
                "name": "index_pgvector",
                "script": "5_index_pgvector.py",
                "description": "Create vector embeddings and indexes",
                "input": "data/normalized/normalized_facts.json",
                "output": "data/embeddings"
            },
            {
                "name": "rag_api",
                "script": "6_rag_api.py",
                "description": "Start RAG API server",
                "input": "data/bridge_table",
                "output": "api_server"
            },
            {
                "name": "dashboard",
                "script": "7_dashboard_app.py",
                "description": "Start Streamlit dashboard",
                "input": "api_server",
                "output": "dashboard"
            }
        ]
        
        # Pipeline configuration
        self.config = {
            "max_documents_per_source": 50,
            "extraction_confidence_threshold": 0.7,
            "normalization_similarity_threshold": 0.8,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_dimension": 384,
            "neo4j_batch_size": 1000,
            "api_port": 8000,
            "dashboard_port": 8501
        }
    
    def run_stage(self, stage_name: str, dry_run: bool = False) -> bool:
        """Run a specific pipeline stage"""
        stage = next((s for s in self.stages if s["name"] == stage_name), None)
        if not stage:
            logger.error(f"Stage '{stage_name}' not found")
            return False
        
        logger.info(f"Running stage: {stage['name']} - {stage['description']}")
        
        if dry_run:
            logger.info(f"DRY RUN: Would execute {stage['script']}")
            return True
        
        try:
            # Build command
            script_path = Path(__file__).parent / "stages" / stage["script"]
            cmd = [sys.executable, str(script_path)]
            
            # Add stage-specific arguments
            if stage_name == "extract_tables":
                cmd.extend(["--pdf-dir", stage["input"]])
                cmd.extend(["--output-dir", stage["output"]])
            elif stage_name == "normalize_schema":
                cmd.extend(["--input-file", stage["input"]])
                cmd.extend(["--output-dir", stage["output"]])
            elif stage_name == "build_fact_table":
                cmd.extend(["--normalized-file", stage["input"]])
                cmd.extend(["--extracted-file", "data/extracted/all_extracted_data.json"])
                cmd.extend(["--output-dir", stage["output"]])
            elif stage_name == "load_neo4j":
                cmd.extend(["--normalized-file", stage["input"]])
                cmd.extend(["--output-dir", stage["output"]])
            elif stage_name == "index_pgvector":
                cmd.extend(["--normalized-file", stage["input"]])
                cmd.extend(["--output-dir", stage["output"]])
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Stage '{stage_name}' completed successfully")
                return True
            else:
                logger.error(f"Stage '{stage_name}' failed: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Stage '{stage_name}' execution failed: {e}")
            return False
    
    def run_pipeline(self, start_stage: str = None, end_stage: str = None, 
                    dry_run: bool = False) -> bool:
        """Run the complete pipeline or a subset"""
        logger.info("Starting AP Education Policy Intelligence Pipeline")
        
        # Determine stages to run
        start_idx = 0
        end_idx = len(self.stages)
        
        if start_stage:
            start_idx = next((i for i, s in enumerate(self.stages) if s["name"] == start_stage), 0)
        
        if end_stage:
            end_idx = next((i for i, s in enumerate(self.stages) if s["name"] == end_stage), len(self.stages)) + 1
        
        stages_to_run = self.stages[start_idx:end_idx]
        
        logger.info(f"Running stages: {[s['name'] for s in stages_to_run]}")
        
        # Run stages sequentially
        success_count = 0
        for stage in stages_to_run:
            if self.run_stage(stage["name"], dry_run):
                success_count += 1
            else:
                logger.error(f"Pipeline failed at stage: {stage['name']}")
                return False
        
        logger.info(f"Pipeline completed successfully: {success_count}/{len(stages_to_run)} stages")
        return True
    
    def validate_prerequisites(self) -> bool:
        """Validate that all prerequisites are met"""
        logger.info("Validating prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False
        
        # Check required directories
        required_dirs = [
            "data/preprocessed/documents",
            "data/extracted",
            "data/normalized",
            "data/bridge_table",
            "data/neo4j",
            "data/embeddings"
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Check for PDF files
        pdf_dir = Path("data/preprocessed/documents")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in data/preprocessed/documents")
            logger.info("Please ensure PDF files are available before running the pipeline")
        
        # Check required Python packages
        required_packages = [
            "pandas", "numpy", "psycopg2", "fastapi", "streamlit",
            "plotly", "requests", "beautifulsoup4", "fitz", "camelot"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.info("Please install missing packages: pip install " + " ".join(missing_packages))
        
        logger.info("Prerequisites validation completed")
        return True
    
    def generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report"""
        report = {
            "pipeline_info": {
                "name": "AP Education Policy Intelligence Pipeline",
                "version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
                "total_stages": len(self.stages)
            },
            "configuration": self.config,
            "stages": self.stages,
            "data_summary": self._get_data_summary(),
            "system_status": self._get_system_status()
        }
        
        return report
    
    def _get_data_summary(self) -> Dict[str, Any]:
        """Get summary of data in each stage"""
        summary = {}
        
        # Check extracted data
        extracted_file = Path("data/extracted/all_extracted_data.json")
        if extracted_file.exists():
            try:
                with open(extracted_file, 'r') as f:
                    extracted_data = json.load(f)
                summary["extracted"] = {
                    "total_pdfs": len(extracted_data),
                    "total_items": sum(len(items) for items in extracted_data.values())
                }
            except:
                summary["extracted"] = {"status": "error"}
        else:
            summary["extracted"] = {"status": "not_found"}
        
        # Check normalized data
        normalized_file = Path("data/normalized/normalized_facts.json")
        if normalized_file.exists():
            try:
                with open(normalized_file, 'r') as f:
                    normalized_data = json.load(f)
                summary["normalized"] = {
                    "total_facts": len(normalized_data)
                }
            except:
                summary["normalized"] = {"status": "error"}
        else:
            summary["normalized"] = {"status": "not_found"}
        
        # Check bridge table
        bridge_files = list(Path("data/bridge_table").glob("*.csv"))
        summary["bridge_table"] = {
            "files": len(bridge_files),
            "status": "available" if bridge_files else "not_found"
        }
        
        # Check Neo4j data
        neo4j_files = list(Path("data/neo4j").glob("*.json"))
        summary["neo4j"] = {
            "files": len(neo4j_files),
            "status": "available" if neo4j_files else "not_found"
        }
        
        # Check embeddings
        embedding_files = list(Path("data/embeddings").glob("*.json"))
        summary["embeddings"] = {
            "files": len(embedding_files),
            "status": "available" if embedding_files else "not_found"
        }
        
        return summary
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        status = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "working_directory": str(Path.cwd()),
            "pipeline_directory": str(Path(__file__).parent),
            "data_directory": str(self.base_dir)
        }
        
        return status
    
    def save_pipeline_report(self, report: Dict[str, Any]):
        """Save pipeline report to file"""
        report_file = self.base_dir / "pipeline_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pipeline report saved to {report_file}")
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate files to save space"""
        logger.info("Cleaning up intermediate files...")
        
        # Files to clean up (keep final outputs)
        cleanup_patterns = [
            "data/extracted/*_extracted.json",  # Keep all_extracted_data.json
            "data/normalized/*.tmp",
            "data/bridge_table/*.tmp",
            "data/neo4j/*.tmp",
            "data/embeddings/*.tmp"
        ]
        
        for pattern in cleanup_patterns:
            for file_path in Path(".").glob(pattern):
                try:
                    file_path.unlink()
                    logger.debug(f"Removed: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='AP Education Policy Intelligence Pipeline')
    parser.add_argument('--start-stage', help='Start from specific stage')
    parser.add_argument('--end-stage', help='End at specific stage')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be executed')
    parser.add_argument('--validate-only', action='store_true', help='Only validate prerequisites')
    parser.add_argument('--report-only', action='store_true', help='Only generate report')
    parser.add_argument('--cleanup', action='store_true', help='Clean up intermediate files')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    try:
        if args.validate_only:
            # Only validate prerequisites
            success = orchestrator.validate_prerequisites()
            sys.exit(0 if success else 1)
        
        elif args.report_only:
            # Only generate report
            report = orchestrator.generate_pipeline_report()
            orchestrator.save_pipeline_report(report)
            print(json.dumps(report, indent=2))
            sys.exit(0)
        
        elif args.cleanup:
            # Only cleanup
            orchestrator.cleanup_intermediate_files()
            sys.exit(0)
        
        else:
            # Run pipeline
            # Validate prerequisites first
            if not orchestrator.validate_prerequisites():
                logger.error("Prerequisites validation failed")
                sys.exit(1)
            
            # Run pipeline
            success = orchestrator.run_pipeline(
                start_stage=args.start_stage,
                end_stage=args.end_stage,
                dry_run=args.dry_run
            )
            
            if success:
                # Generate final report
                report = orchestrator.generate_pipeline_report()
                orchestrator.save_pipeline_report(report)
                
                logger.info("Pipeline completed successfully!")
                print("\n" + "="*60)
                print("PIPELINE COMPLETED SUCCESSFULLY")
                print("="*60)
                print(f"Total stages: {len(orchestrator.stages)}")
                print(f"Data directory: {orchestrator.base_dir}")
                print(f"Report saved: {orchestrator.base_dir}/pipeline_report.json")
                print("="*60)
            else:
                logger.error("Pipeline failed")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
