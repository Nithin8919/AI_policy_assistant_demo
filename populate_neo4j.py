#!/usr/bin/env python3
"""
Populate Neo4j Database with AP Education Graph Data
Creates nodes and relationships for education policy knowledge graph
"""
import os
import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection details (use env vars with sensible defaults)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class Neo4jPopulator:
    """Populates Neo4j with AP education policy data"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        logger.info(f"Connected to Neo4j at {NEO4J_URI}")
    
    def clear_database(self):
        """Clear all existing data"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing database")
    
    def create_constraints(self):
        """Create unique constraints"""
        constraints = [
            "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT district_id IF NOT EXISTS FOR (d:District) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT indicator_id IF NOT EXISTS FOR (i:Indicator) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (doc:Document) REQUIRE doc.id IS UNIQUE",
            "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (p:Policy) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT year_id IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.debug(f"Constraint might already exist: {e}")
    
    def create_districts(self):
        """Create AP district nodes"""
        districts = [
            {"id": "anantapur", "name": "Anantapur", "population": 4081148, "literacy_rate": 63.57},
            {"id": "chittoor", "name": "Chittoor", "population": 4174064, "literacy_rate": 71.53},
            {"id": "east_godavari", "name": "East Godavari", "population": 5154296, "literacy_rate": 71.35},
            {"id": "guntur", "name": "Guntur", "population": 4887813, "literacy_rate": 67.40},
            {"id": "krishna", "name": "Krishna", "population": 4517398, "literacy_rate": 73.74},
            {"id": "kurnool", "name": "Kurnool", "population": 4046601, "literacy_rate": 59.97},
            {"id": "nellore", "name": "Nellore", "population": 2963557, "literacy_rate": 68.90},
            {"id": "prakasam", "name": "Prakasam", "population": 3392764, "literacy_rate": 63.08},
            {"id": "srikakulam", "name": "Srikakulam", "population": 2703114, "literacy_rate": 61.74},
            {"id": "visakhapatnam", "name": "Visakhapatnam", "population": 4290589, "literacy_rate": 67.22},
            {"id": "vizianagaram", "name": "Vizianagaram", "population": 2344474, "literacy_rate": 58.89},
            {"id": "west_godavari", "name": "West Godavari", "population": 3934782, "literacy_rate": 74.63},
            {"id": "kadapa", "name": "Kadapa", "population": 2884524, "literacy_rate": 68.36}
        ]
        
        with self.driver.session() as session:
            for district in districts:
                session.run("""
                    CREATE (d:District {
                        id: $id,
                        name: $name,
                        population: $population,
                        literacy_rate: $literacy_rate,
                        created_at: datetime()
                    })
                """, **district)
                logger.info(f"Created district: {district['name']}")
    
    def create_indicators(self):
        """Create education indicator nodes"""
        indicators = [
            {"id": "enrollment", "name": "Student Enrollment", "category": "Students", "description": "Total number of students enrolled"},
            {"id": "teachers", "name": "Teachers", "category": "Staff", "description": "Total number of teaching staff"},
            {"id": "schools", "name": "Schools", "category": "Infrastructure", "description": "Total number of educational institutions"},
            {"id": "dropout_rate", "name": "Dropout Rate", "category": "Performance", "description": "Percentage of students dropping out"},
            {"id": "ptr", "name": "Pupil Teacher Ratio", "category": "Performance", "description": "Ratio of students to teachers"},
            {"id": "infrastructure", "name": "Infrastructure", "category": "Infrastructure", "description": "School buildings and facilities"},
            {"id": "budget", "name": "Education Budget", "category": "Finance", "description": "Education sector budget allocation"},
            {"id": "teacher_training", "name": "Teacher Training", "category": "Staff", "description": "Teacher professional development programs"},
            {"id": "digital_education", "name": "Digital Education", "category": "Technology", "description": "ICT and digital learning initiatives"},
            {"id": "mid_day_meal", "name": "Mid Day Meal", "category": "Welfare", "description": "School nutrition program"}
        ]
        
        with self.driver.session() as session:
            for indicator in indicators:
                session.run("""
                    CREATE (i:Indicator {
                        id: $id,
                        name: $name,
                        category: $category,
                        description: $description,
                        created_at: datetime()
                    })
                """, **indicator)
                logger.info(f"Created indicator: {indicator['name']}")
    
    def create_policies(self):
        """Create education policy nodes"""
        policies = [
            {
                "id": "rte_act_2009",
                "name": "Right to Education Act 2009",
                "type": "Central Act",
                "description": "Free and compulsory education for children aged 6-14",
                "year_enacted": 2009
            },
            {
                "id": "ap_education_policy_2020",
                "name": "AP Education Policy 2020",
                "type": "State Policy",
                "description": "Comprehensive education reform policy for Andhra Pradesh",
                "year_enacted": 2020
            },
            {
                "id": "teacher_training_policy",
                "name": "Teacher Training and Development Policy",
                "type": "State Policy", 
                "description": "Policy for continuous professional development of teachers",
                "year_enacted": 2021
            },
            {
                "id": "digital_ap_2029",
                "name": "Digital AP 2029",
                "type": "Digital Policy",
                "description": "Digital transformation roadmap including education sector",
                "year_enacted": 2019
            }
        ]
        
        with self.driver.session() as session:
            for policy in policies:
                session.run("""
                    CREATE (p:Policy {
                        id: $id,
                        name: $name,
                        type: $type,
                        description: $description,
                        year_enacted: $year_enacted,
                        created_at: datetime()
                    })
                """, **policy)
                logger.info(f"Created policy: {policy['name']}")
    
    def create_facts_with_relationships(self):
        """Create fact nodes and their relationships"""
        facts_data = [
            # Teacher Training Facts
            {
                "fact_id": "teacher_training_krishna_2023",
                "indicator_id": "teacher_training",
                "district_id": "krishna",
                "year": 2023,
                "value": 850,
                "unit": "teachers_trained",
                "content": "850 teachers completed professional development training in Krishna district during 2023",
                "source": "CSE_Training_Report_2023"
            },
            {
                "fact_id": "teacher_training_guntur_2023",
                "indicator_id": "teacher_training", 
                "district_id": "guntur",
                "year": 2023,
                "value": 720,
                "unit": "teachers_trained",
                "content": "720 teachers participated in capacity building programs in Guntur district in 2023",
                "source": "SCERT_Annual_Report_2023"
            },
            {
                "fact_id": "digital_training_visakhapatnam_2023",
                "indicator_id": "digital_education",
                "district_id": "visakhapatnam",
                "year": 2023,
                "value": 95,
                "unit": "percentage",
                "content": "95% of teachers in Visakhapatnam completed digital education training modules",
                "source": "AP_Digital_Education_2023"
            },
            # General Education Facts
            {
                "fact_id": "enrollment_krishna_2023",
                "indicator_id": "enrollment",
                "district_id": "krishna",
                "year": 2023,
                "value": 125000,
                "unit": "students",
                "content": "Total student enrollment in Krishna district was 125,000 in 2023",
                "source": "UDISE_Plus_2023"
            },
            {
                "fact_id": "teachers_total_ap_2023",
                "indicator_id": "teachers",
                "district_id": None,  # State level
                "year": 2023,
                "value": 180000,
                "unit": "teachers",
                "content": "Andhra Pradesh has 180,000 teachers across all districts in 2023",
                "source": "AP_Teacher_Census_2023"
            },
            {
                "fact_id": "ptr_guntur_2023",
                "indicator_id": "ptr",
                "district_id": "guntur",
                "year": 2023,
                "value": 22,
                "unit": "ratio",
                "content": "Pupil-Teacher Ratio in Guntur district is 22:1 in academic year 2023",
                "source": "Education_Statistics_2023"
            }
        ]
        
        with self.driver.session() as session:
            for fact_data in facts_data:
                # Create fact node
                session.run("""
                    CREATE (f:Fact {
                        id: $fact_id,
                        year: $year,
                        value: $value,
                        unit: $unit,
                        content: $content,
                        source: $source,
                        created_at: datetime()
                    })
                """, **fact_data)
                
                # Create relationship with indicator
                session.run("""
                    MATCH (f:Fact {id: $fact_id})
                    MATCH (i:Indicator {id: $indicator_id})
                    CREATE (f)-[:MEASURES]->(i)
                """, fact_id=fact_data["fact_id"], indicator_id=fact_data["indicator_id"])
                
                # Create relationship with district (if specified)
                if fact_data["district_id"]:
                    session.run("""
                        MATCH (f:Fact {id: $fact_id})
                        MATCH (d:District {id: $district_id})
                        CREATE (f)-[:LOCATED_IN]->(d)
                    """, fact_id=fact_data["fact_id"], district_id=fact_data["district_id"])
                
                logger.info(f"Created fact: {fact_data['fact_id']}")
    
    def create_policy_relationships(self):
        """Create relationships between policies and indicators/districts"""
        relationships = [
            # Teacher Training Policy affects Teacher Training indicator
            {
                "policy_id": "teacher_training_policy",
                "indicator_id": "teacher_training",
                "relationship": "GOVERNS"
            },
            # RTE Act affects multiple indicators
            {
                "policy_id": "rte_act_2009",
                "indicator_id": "enrollment",
                "relationship": "MANDATES"
            },
            {
                "policy_id": "rte_act_2009", 
                "indicator_id": "infrastructure",
                "relationship": "REQUIRES"
            },
            # Digital AP affects digital education
            {
                "policy_id": "digital_ap_2029",
                "indicator_id": "digital_education",
                "relationship": "PROMOTES"
            }
        ]
        
        with self.driver.session() as session:
            for rel in relationships:
                session.run(f"""
                    MATCH (p:Policy {{id: $policy_id}})
                    MATCH (i:Indicator {{id: $indicator_id}})
                    CREATE (p)-[:{rel['relationship']}]->(i)
                """, policy_id=rel["policy_id"], indicator_id=rel["indicator_id"])
                logger.info(f"Created relationship: {rel['policy_id']} {rel['relationship']} {rel['indicator_id']}")
    
    def create_cross_indicator_relationships(self):
        """Create relationships between related indicators"""
        relationships = [
            ("teacher_training", "teachers", "IMPROVES"),
            ("teacher_training", "ptr", "AFFECTS"),
            ("digital_education", "teacher_training", "PART_OF"),
            ("enrollment", "ptr", "DETERMINES"),
            ("infrastructure", "enrollment", "SUPPORTS"),
            ("budget", "teacher_training", "FUNDS"),
            ("budget", "infrastructure", "FUNDS")
        ]
        
        with self.driver.session() as session:
            for source, target, relationship in relationships:
                session.run(f"""
                    MATCH (s:Indicator {{id: $source}})
                    MATCH (t:Indicator {{id: $target}})
                    CREATE (s)-[:{relationship}]->(t)
                """, source=source, target=target)
                logger.info(f"Created indicator relationship: {source} {relationship} {target}")
    
    def populate_all(self):
        """Populate the entire knowledge graph"""
        logger.info("Starting Neo4j database population...")
        
        # Clear existing data
        self.clear_database()
        
        # Create constraints
        self.create_constraints()
        
        # Create nodes
        self.create_districts()
        self.create_indicators() 
        self.create_policies()
        
        # Create facts and relationships
        self.create_facts_with_relationships()
        self.create_policy_relationships()
        self.create_cross_indicator_relationships()
        
        logger.info("Neo4j database population completed!")
    
    def get_stats(self):
        """Get database statistics"""
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes by label
            labels = ["District", "Indicator", "Policy", "Fact"]
            for label in labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                stats[f"{label.lower()}_count"] = result.single()["count"]
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats["relationship_count"] = result.single()["count"]
            
            return stats
    
    def test_teacher_training_queries(self):
        """Test queries related to teacher training"""
        with self.driver.session() as session:
            logger.info("Testing teacher training queries...")
            
            # Query 1: Find all teacher training facts
            result = session.run("""
                MATCH (f:Fact)-[:MEASURES]->(i:Indicator {id: 'teacher_training'})
                OPTIONAL MATCH (f)-[:LOCATED_IN]->(d:District)
                RETURN f.id, f.content, f.value, f.unit, d.name as district
                ORDER BY f.value DESC
            """)
            
            print("\n=== Teacher Training Facts ===")
            for record in result:
                print(f"ID: {record['f.id']}")
                print(f"District: {record['district'] or 'State Level'}")
                print(f"Value: {record['f.value']} {record['f.unit']}")
                print(f"Content: {record['f.content']}")
                print("-" * 50)
            
            # Query 2: Find policies affecting teacher training
            result = session.run("""
                MATCH (p:Policy)-[r]->(i:Indicator {id: 'teacher_training'})
                RETURN p.name, p.description, type(r) as relationship
            """)
            
            print("\n=== Policies Affecting Teacher Training ===")
            for record in result:
                print(f"Policy: {record['p.name']}")
                print(f"Relationship: {record['relationship']}")
                print(f"Description: {record['p.description']}")
                print("-" * 50)
            
            # Query 3: Find related indicators
            result = session.run("""
                MATCH (i1:Indicator {id: 'teacher_training'})-[r]-(i2:Indicator)
                RETURN i2.name, i2.category, type(r) as relationship
            """)
            
            print("\n=== Related Indicators ===")
            for record in result:
                print(f"Indicator: {record['i2.name']}")
                print(f"Category: {record['i2.category']}")
                print(f"Relationship: {record['relationship']}")
                print("-" * 50)
    
    def close(self):
        """Close the database connection"""
        self.driver.close()

if __name__ == "__main__":
    populator = Neo4jPopulator()
    
    try:
        # Populate database
        populator.populate_all()
        
        # Show statistics
        stats = populator.get_stats()
        print(f"\n=== Database Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Test teacher training queries
        populator.test_teacher_training_queries()
        
    except Exception as e:
        logger.error(f"Error during population: {e}")
    finally:
        populator.close()