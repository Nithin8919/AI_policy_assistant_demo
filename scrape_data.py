#!/usr/bin/env python3
"""
Data Scraping Pipeline Orchestrator
Runs all scrapers and organizes data for preprocessing
"""
import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_pipeline.scrapers.go_scraper import GOScraper
from data_pipeline.scrapers.cse_scraper import CSEPortalScraper
from data_pipeline.scrapers.scert_scraper import SCERTScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataScrapingPipeline:
    """Main orchestrator for data scraping"""
    
    def __init__(self, output_base_dir: str = "data/raw"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scrapers
        self.go_scraper = GOScraper(output_dir=str(self.output_base_dir / "gos"))
        self.cse_scraper = CSEPortalScraper(output_dir=str(self.output_base_dir / "cse"))
        self.scert_scraper = SCERTScraper(output_dir=str(self.output_base_dir / "scert"))
        
        # Results storage
        self.all_results = {
            'go_documents': [],
            'cse_documents': [],
            'scert_documents': [],
            'scraping_summary': {}
        }
    
    def run_full_scraping(self, max_docs_per_source: int = 50) -> Dict[str, Any]:
        """
        Run complete data scraping pipeline
        
        Args:
            max_docs_per_source: Maximum documents per data source
            
        Returns:
            Complete scraping results
        """
        logger.info("Starting full data scraping pipeline")
        start_time = datetime.now()
        
        try:
            # 1. Scrape Government Orders
            logger.info("=" * 50)
            logger.info("SCRAPING GOVERNMENT ORDERS")
            logger.info("=" * 50)
            go_docs = self.go_scraper.scrape_gos(
                pages=10, 
                max_documents=max_docs_per_source
            )
            self.all_results['go_documents'] = go_docs
            logger.info(f"✅ Scraped {len(go_docs)} Government Orders")
            
            # 2. Scrape CSE Portal
            logger.info("=" * 50)
            logger.info("SCRAPING CSE PORTAL")
            logger.info("=" * 50)
            cse_docs = self.cse_scraper.scrape_circulars(
                max_documents=max_docs_per_source
            )
            self.all_results['cse_documents'] = cse_docs
            logger.info(f"✅ Scraped {len(cse_docs)} CSE Portal documents")
            
            # 3. Scrape SCERT Materials
            logger.info("=" * 50)
            logger.info("SCRAPING SCERT MATERIALS")
            logger.info("=" * 50)
            scert_docs = self.scert_scraper.scrape_curriculum_materials(
                max_documents=max_docs_per_source
            )
            self.all_results['scert_documents'] = scert_docs
            logger.info(f"✅ Scraped {len(scert_docs)} SCERT documents")
            
            # 4. Generate summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.all_results['scraping_summary'] = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'total_documents': len(go_docs) + len(cse_docs) + len(scert_docs),
                'go_count': len(go_docs),
                'cse_count': len(cse_docs),
                'scert_count': len(scert_docs),
                'status': 'completed'
            }
            
            # 5. Save results
            self._save_scraping_results()
            self._organize_data_for_preprocessing()
            
            logger.info("=" * 50)
            logger.info("SCRAPING COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            logger.info(f"Total documents scraped: {self.all_results['scraping_summary']['total_documents']}")
            logger.info(f"Duration: {duration}")
            logger.info(f"Results saved to: {self.output_base_dir}")
            
            return self.all_results
            
        except Exception as e:
            logger.error(f"Scraping pipeline failed: {e}")
            self.all_results['scraping_summary']['status'] = 'failed'
            self.all_results['scraping_summary']['error'] = str(e)
            raise
    
    def run_quick_scraping(self, max_docs_per_source: int = 10) -> Dict[str, Any]:
        """
        Run quick data scraping for testing
        
        Args:
            max_docs_per_source: Maximum documents per data source
            
        Returns:
            Quick scraping results
        """
        logger.info("Starting quick data scraping (testing mode)")
        
        try:
            # Quick GO scraping
            go_docs = self.go_scraper.scrape_gos(pages=2, max_documents=max_docs_per_source)
            self.all_results['go_documents'] = go_docs
            
            # Quick CSE scraping
            cse_docs = self.cse_scraper.scrape_circulars(max_documents=max_docs_per_source)
            self.all_results['cse_documents'] = cse_docs
            
            # Quick SCERT scraping
            scert_docs = self.scert_scraper.scrape_curriculum_materials(max_documents=max_docs_per_source)
            self.all_results['scert_documents'] = scert_docs
            
            # Save results
            self._save_scraping_results()
            self._organize_data_for_preprocessing()
            
            logger.info(f"Quick scraping completed: {len(go_docs + cse_docs + scert_docs)} documents")
            return self.all_results
            
        except Exception as e:
            logger.error(f"Quick scraping failed: {e}")
            raise
    
    def _save_scraping_results(self):
        """Save all scraping results to JSON files"""
        try:
            # Save individual scraper results
            self.go_scraper.save_metadata(self.all_results['go_documents'])
            self.cse_scraper.save_metadata(self.all_results['cse_documents'])
            self.scert_scraper.save_metadata(self.all_results['scert_documents'])
            
            # Save combined results
            combined_file = self.output_base_dir / "scraping_results.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved combined results to {combined_file}")
            
        except Exception as e:
            logger.error(f"Failed to save scraping results: {e}")
    
    def _organize_data_for_preprocessing(self):
        """Organize scraped data for preprocessing pipeline"""
        try:
            # Create preprocessing directory structure
            preprocess_dir = self.output_base_dir.parent / "preprocessed"
            preprocess_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (preprocess_dir / "documents").mkdir(exist_ok=True)
            (preprocess_dir / "metadata").mkdir(exist_ok=True)
            (preprocess_dir / "text").mkdir(exist_ok=True)
            
            # Copy documents to preprocessing directory
            all_docs = (
                self.all_results['go_documents'] + 
                self.all_results['cse_documents'] + 
                self.all_results['scert_documents']
            )
            
            copied_count = 0
            for doc in all_docs:
                try:
                    source_path = Path(doc['file_path'])
                    if source_path.exists():
                        # Copy to preprocessing directory
                        dest_path = preprocess_dir / "documents" / doc['filename']
                        dest_path.write_bytes(source_path.read_bytes())
                        copied_count += 1
                        
                except Exception as e:
                    logger.debug(f"Failed to copy {doc['filename']}: {e}")
                    continue
            
            # Save organized metadata
            organized_metadata = {
                'total_documents': len(all_docs),
                'copied_documents': copied_count,
                'document_types': {
                    'go': len(self.all_results['go_documents']),
                    'cse': len(self.all_results['cse_documents']),
                    'scert': len(self.all_results['scert_documents'])
                },
                'documents': all_docs,
                'preprocessing_ready': True,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_file = preprocess_dir / "metadata" / "organized_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(organized_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Organized {copied_count} documents for preprocessing")
            logger.info(f"Preprocessing data ready in: {preprocess_dir}")
            
        except Exception as e:
            logger.error(f"Failed to organize data for preprocessing: {e}")
    
    def get_scraping_summary(self) -> Dict[str, Any]:
        """Get summary of scraping results"""
        return self.all_results.get('scraping_summary', {})
    
    def list_scraped_files(self) -> List[str]:
        """List all scraped files"""
        all_files = []
        
        for doc_type in ['go_documents', 'cse_documents', 'scert_documents']:
            for doc in self.all_results.get(doc_type, []):
                if 'file_path' in doc:
                    all_files.append(doc['file_path'])
        
        return all_files

def main():
    """Main function to run scraping pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Scraping Pipeline')
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                       help='Scraping mode: full or quick')
    parser.add_argument('--max-docs', type=int, default=20,
                       help='Maximum documents per source')
    parser.add_argument('--output-dir', default='data/raw',
                       help='Output directory for scraped data')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DataScrapingPipeline(output_base_dir=args.output_dir)
    
    try:
        if args.mode == 'full':
            results = pipeline.run_full_scraping(max_docs_per_source=args.max_docs)
        else:
            results = pipeline.run_quick_scraping(max_docs_per_source=args.max_docs)
        
        # Print summary
        summary = pipeline.get_scraping_summary()
        print("\n" + "="*60)
        print("SCRAPING SUMMARY")
        print("="*60)
        print(f"Total Documents: {summary.get('total_documents', 0)}")
        print(f"GO Documents: {summary.get('go_count', 0)}")
        print(f"CSE Documents: {summary.get('cse_count', 0)}")
        print(f"SCERT Documents: {summary.get('scert_count', 0)}")
        print(f"Status: {summary.get('status', 'unknown')}")
        print("="*60)
        
        # List some files
        files = pipeline.list_scraped_files()
        if files:
            print(f"\nScraped Files ({len(files)} total):")
            for i, file_path in enumerate(files[:10]):  # Show first 10
                print(f"  {i+1}. {Path(file_path).name}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
        
        print(f"\nData ready for preprocessing in: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
