#!/usr/bin/env python3
"""
Mock Data Generator for Education Policy Documents
Generates realistic sample data for testing the preprocessing pipeline
"""
import os
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MockDataGenerator:
    """Generate realistic mock education policy documents"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample document templates
        self.go_templates = [
            "Implementation of National Education Policy 2020 in Andhra Pradesh",
            "Guidelines for School Infrastructure Development",
            "Teacher Recruitment and Training Policy",
            "Student Assessment and Evaluation Framework",
            "Digital Learning Initiative for Schools",
            "Midday Meal Program Enhancement",
            "School Safety and Security Measures",
            "Curriculum Framework for Primary Education",
            "Special Education Needs Support Program",
            "School Health and Wellness Program"
        ]
        
        self.cse_templates = [
            "Circular on Examination Schedule and Guidelines",
            "Teacher Professional Development Program",
            "Student Enrollment and Admission Process",
            "School Quality Assessment Framework",
            "Digital Learning Resources Distribution",
            "Academic Calendar and Timetable Guidelines",
            "Student Scholarship and Financial Aid Program",
            "School Infrastructure Maintenance Guidelines",
            "Teacher Performance Evaluation System",
            "Student Health and Nutrition Program"
        ]
        
        self.scert_templates = [
            "Primary School Curriculum Framework",
            "Teacher Training Manual for Mathematics",
            "Assessment Tools for Language Learning",
            "Science Education Resource Materials",
            "Social Studies Teaching Guidelines",
            "Art and Craft Education Framework",
            "Physical Education Curriculum Standards",
            "Environmental Education Program",
            "Digital Literacy Training Materials",
            "Inclusive Education Resource Guide"
        ]
        
        # Sample content templates
        self.content_templates = {
            "policy": """
            POLICY FRAMEWORK FOR EDUCATION DEVELOPMENT
            
            1. OBJECTIVES
            The primary objective of this policy is to enhance the quality of education 
            and ensure equitable access to learning opportunities for all students.
            
            2. IMPLEMENTATION STRATEGY
            - Capacity building for teachers and administrators
            - Infrastructure development and modernization
            - Curriculum enhancement and digital integration
            - Student support and welfare programs
            
            3. MONITORING AND EVALUATION
            Regular assessment and monitoring mechanisms will be established to ensure 
            effective implementation and continuous improvement.
            
            4. STAKEHOLDER ENGAGEMENT
            Collaboration with parents, teachers, students, and community members is 
            essential for successful policy implementation.
            """,
            
            "circular": """
            CIRCULAR: IMPLEMENTATION GUIDELINES
            
            To: All District Education Officers, Principals, and Teachers
            
            Subject: Implementation of New Educational Initiatives
            
            Dear Colleagues,
            
            This circular outlines the implementation guidelines for the new educational 
            initiatives approved by the government. Please ensure compliance with the 
            following directives:
            
            1. IMMEDIATE ACTIONS REQUIRED
            - Review and update school policies
            - Conduct staff training sessions
            - Prepare implementation reports
            
            2. TIMELINE FOR IMPLEMENTATION
            Phase 1: Planning and Preparation (Month 1-2)
            Phase 2: Implementation (Month 3-6)
            Phase 3: Monitoring and Evaluation (Month 7-12)
            
            3. SUPPORT AND RESOURCES
            Technical support and necessary resources will be provided through the 
            respective departments.
            
            For queries, contact the Education Department.
            
            Best regards,
            Director of Education
            """,
            
            "curriculum": """
            CURRICULUM FRAMEWORK DOCUMENT
            
            LEARNING OBJECTIVES
            Students will develop:
            - Critical thinking and problem-solving skills
            - Communication and collaboration abilities
            - Digital literacy and technological competence
            - Cultural awareness and social responsibility
            
            SUBJECT AREAS
            1. Language and Literature
               - Reading comprehension and analysis
               - Creative writing and expression
               - Communication skills development
            
            2. Mathematics and Science
               - Numerical reasoning and computation
               - Scientific inquiry and experimentation
               - Data analysis and interpretation
            
            3. Social Studies
               - Historical understanding and analysis
               - Geographic knowledge and skills
               - Civic responsibility and citizenship
            
            ASSESSMENT FRAMEWORK
            - Formative assessment for continuous improvement
            - Summative assessment for comprehensive evaluation
            - Portfolio-based assessment for holistic development
            
            TEACHING METHODOLOGIES
            - Student-centered learning approaches
            - Collaborative and cooperative learning
            - Technology-integrated instruction
            - Differentiated instruction for diverse learners
            """
        }
    
    def generate_go_documents(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate mock Government Order documents"""
        documents = []
        
        for i in range(count):
            template = random.choice(self.go_templates)
            go_number = f"G.O.Ms.No.{random.randint(100, 999)}"
            
            # Generate document content
            content = self._generate_document_content("policy")
            
            # Create document metadata
            doc_metadata = {
                'document_id': f"GO_{go_number.split('.')[-1]}",
                'title': template,
                'go_number': go_number,
                'source_url': f"https://goir.ap.gov.in/orders/{go_number.replace('.', '_')}.pdf",
                'filename': f"GO_{go_number.split('.')[-1]}.pdf",
                'file_path': str(self.output_dir / "gos" / f"GO_{go_number.split('.')[-1]}.pdf"),
                'document_type': 'GO',
                'scraped_at': datetime.now().timestamp(),
                'status': 'generated',
                'content': content,
                'word_count': len(content.split()),
                'page_count': random.randint(3, 15)
            }
            
            documents.append(doc_metadata)
        
        return documents
    
    def generate_cse_documents(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate mock CSE Portal documents"""
        documents = []
        
        for i in range(count):
            template = random.choice(self.cse_templates)
            circular_number = f"CSE/{random.randint(1000, 9999)}/{datetime.now().year}"
            
            # Generate document content
            content = self._generate_document_content("circular")
            
            # Create document metadata
            doc_metadata = {
                'document_id': f"CSE_{circular_number.split('/')[1]}",
                'title': template,
                'circular_number': circular_number,
                'source_url': f"https://cse.ap.gov.in/circulars/{circular_number.replace('/', '_')}.pdf",
                'filename': f"CSE_{circular_number.split('/')[1]}.pdf",
                'file_path': str(self.output_dir / "cse" / f"CSE_{circular_number.split('/')[1]}.pdf"),
                'document_type': 'CSE_CIRCULAR',
                'scraped_at': datetime.now().timestamp(),
                'status': 'generated',
                'content': content,
                'word_count': len(content.split()),
                'page_count': random.randint(2, 10)
            }
            
            documents.append(doc_metadata)
        
        return documents
    
    def generate_scert_documents(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate mock SCERT documents"""
        documents = []
        
        for i in range(count):
            template = random.choice(self.scert_templates)
            doc_number = f"SCERT/{random.randint(100, 999)}/{datetime.now().year}"
            doc_type = random.choice(['TEXTBOOK', 'CURRICULUM', 'TRAINING_MATERIAL', 'ASSESSMENT'])
            
            # Generate document content
            content = self._generate_document_content("curriculum")
            
            # Create document metadata
            doc_metadata = {
                'document_id': f"SCERT_{doc_number.split('/')[1]}",
                'title': template,
                'document_number': doc_number,
                'document_type': doc_type,
                'section': random.choice(['curriculum', 'textbooks', 'training-materials']),
                'source_url': f"https://apscert.gov.in/materials/{doc_number.replace('/', '_')}.pdf",
                'filename': f"SCERT_{doc_type}_{doc_number.split('/')[1]}.pdf",
                'file_path': str(self.output_dir / "scert" / f"SCERT_{doc_type}_{doc_number.split('/')[1]}.pdf"),
                'scraped_at': datetime.now().timestamp(),
                'status': 'generated',
                'content': content,
                'word_count': len(content.split()),
                'page_count': random.randint(5, 25)
            }
            
            documents.append(doc_metadata)
        
        return documents
    
    def _generate_document_content(self, content_type: str) -> str:
        """Generate document content based on type"""
        base_content = self.content_templates.get(content_type, self.content_templates["policy"])
        
        # Add some variation
        variations = [
            "\n\nADDITIONAL GUIDELINES:\n- Ensure compliance with state regulations\n- Maintain proper documentation\n- Regular monitoring and reporting required",
            "\n\nIMPLEMENTATION TIMELINE:\n- Phase 1: Planning (Months 1-2)\n- Phase 2: Implementation (Months 3-6)\n- Phase 3: Evaluation (Months 7-12)",
            "\n\nRESOURCE REQUIREMENTS:\n- Human resources: Trained personnel\n- Infrastructure: Modern facilities\n- Technology: Digital tools and platforms",
            "\n\nEXPECTED OUTCOMES:\n- Improved learning outcomes\n- Enhanced teacher effectiveness\n- Better student engagement\n- Increased parent satisfaction"
        ]
        
        return base_content + random.choice(variations)
    
    def save_documents_as_text(self, documents: List[Dict[str, Any]], subdir: str):
        """Save documents as text files for processing"""
        subdir_path = self.output_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        
        for doc in documents:
            if 'content' in doc:
                text_file = subdir_path / f"{doc['filename'].replace('.pdf', '.txt')}"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"TITLE: {doc['title']}\n\n")
                    f.write(f"DOCUMENT TYPE: {doc['document_type']}\n")
                    f.write(f"DOCUMENT NUMBER: {doc.get('go_number', doc.get('circular_number', doc.get('document_number', 'N/A')))}\n")
                    f.write(f"SOURCE: {doc['source_url']}\n")
                    f.write(f"GENERATED: {datetime.fromtimestamp(doc['scraped_at']).isoformat()}\n")
                    f.write("="*80 + "\n\n")
                    f.write(doc['content'])
                
                # Update file path to point to text file
                doc['file_path'] = str(text_file)
                doc['filename'] = text_file.name
    
    def generate_all_data(self, count_per_source: int = 20) -> Dict[str, Any]:
        """Generate all mock data"""
        logger.info("Generating mock education policy documents")
        
        # Generate documents
        go_docs = self.generate_go_documents(count_per_source)
        cse_docs = self.generate_cse_documents(count_per_source)
        scert_docs = self.generate_scert_documents(count_per_source)
        
        # Save as text files
        self.save_documents_as_text(go_docs, "gos")
        self.save_documents_as_text(cse_docs, "cse")
        self.save_documents_as_text(scert_docs, "scert")
        
        # Save metadata
        all_results = {
            'go_documents': go_docs,
            'cse_documents': cse_docs,
            'scert_documents': scert_docs,
            'scraping_summary': {
                'start_time': datetime.now().isoformat(),
                'total_documents': len(go_docs) + len(cse_docs) + len(scert_docs),
                'go_count': len(go_docs),
                'cse_count': len(cse_docs),
                'scert_count': len(scert_docs),
                'status': 'generated',
                'data_type': 'mock'
            }
        }
        
        # Save individual metadata files
        (self.output_dir / "gos").mkdir(exist_ok=True)
        (self.output_dir / "cse").mkdir(exist_ok=True)
        (self.output_dir / "scert").mkdir(exist_ok=True)
        
        with open(self.output_dir / "gos" / "go_metadata.json", 'w') as f:
            json.dump(go_docs, f, indent=2)
        
        with open(self.output_dir / "cse" / "cse_metadata.json", 'w') as f:
            json.dump(cse_docs, f, indent=2)
        
        with open(self.output_dir / "scert" / "scert_metadata.json", 'w') as f:
            json.dump(scert_docs, f, indent=2)
        
        # Save combined results
        with open(self.output_dir / "scraping_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Organize for preprocessing
        self._organize_for_preprocessing(all_results)
        
        logger.info(f"Generated {all_results['scraping_summary']['total_documents']} mock documents")
        return all_results
    
    def _organize_for_preprocessing(self, results: Dict[str, Any]):
        """Organize generated data for preprocessing"""
        preprocess_dir = self.output_dir.parent / "preprocessed"
        preprocess_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (preprocess_dir / "documents").mkdir(exist_ok=True)
        (preprocess_dir / "metadata").mkdir(exist_ok=True)
        (preprocess_dir / "text").mkdir(exist_ok=True)
        
        # Copy text files to preprocessing directory
        all_docs = results['go_documents'] + results['cse_documents'] + results['scert_documents']
        
        for doc in all_docs:
            source_path = Path(doc['file_path'])
            if source_path.exists():
                dest_path = preprocess_dir / "documents" / doc['filename']
                dest_path.write_text(source_path.read_text(), encoding='utf-8')
        
        # Save organized metadata
        organized_metadata = {
            'total_documents': len(all_docs),
            'document_types': {
                'go': len(results['go_documents']),
                'cse': len(results['cse_documents']),
                'scert': len(results['scert_documents'])
            },
            'documents': all_docs,
            'preprocessing_ready': True,
            'created_at': datetime.now().isoformat(),
            'data_type': 'mock'
        }
        
        metadata_file = preprocess_dir / "metadata" / "organized_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(organized_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Organized {len(all_docs)} documents for preprocessing")

def main():
    """Main function to generate mock data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Mock Education Policy Data')
    parser.add_argument('--count', type=int, default=20,
                       help='Number of documents per source')
    parser.add_argument('--output-dir', default='data/raw',
                       help='Output directory for generated data')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate data
    generator = MockDataGenerator(output_dir=args.output_dir)
    results = generator.generate_all_data(count_per_source=args.count)
    
    # Print summary
    summary = results['scraping_summary']
    print("\n" + "="*60)
    print("MOCK DATA GENERATION SUMMARY")
    print("="*60)
    print(f"Total Documents: {summary['total_documents']}")
    print(f"GO Documents: {summary['go_count']}")
    print(f"CSE Documents: {summary['cse_count']}")
    print(f"SCERT Documents: {summary['scert_count']}")
    print(f"Status: {summary['status']}")
    print("="*60)
    
    print(f"\nMock data generated in: {args.output_dir}")
    print(f"Preprocessing data ready in: {Path(args.output_dir).parent / 'preprocessed'}")

if __name__ == "__main__":
    main()
