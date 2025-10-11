// Neo4j Ontology Schema for Andhra Pradesh Education Policy Knowledge Graph
// This file defines the node types, relationships, and constraints for the knowledge graph

// Create constraints for unique identifiers
CREATE CONSTRAINT FOR (a:Act) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT FOR (r:Rule) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT FOR (j:Judgment) REQUIRE j.id IS UNIQUE;
CREATE CONSTRAINT FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT FOR (p:Policy) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT FOR (o:Organization) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT FOR (per:Person) REQUIRE per.id IS UNIQUE;
CREATE CONSTRAINT FOR (loc:Location) REQUIRE loc.id IS UNIQUE;
CREATE CONSTRAINT FOR (sch:School) REQUIRE sch.id IS UNIQUE;
CREATE CONSTRAINT FOR (cur:Curriculum) REQUIRE cur.id IS UNIQUE;
CREATE CONSTRAINT FOR (ass:Assessment) REQUIRE ass.id IS UNIQUE;

// Create indexes for better query performance
CREATE INDEX FOR (d:Document) ON (d.document_type);
CREATE INDEX FOR (d:Document) ON (d.source_url);
CREATE INDEX FOR (e:Entity) ON (e.label);
CREATE INDEX FOR (e:Entity) ON (e.confidence);
CREATE INDEX FOR (r:RELATION) ON (r.type);

// Define node types and their properties

// Document nodes represent policy documents, GOs, circulars, etc.
// Properties: id, name, document_type, source_url, text_length, word_count, page_count, language, processing_date
CREATE (d:Document:Entity {
    id: 'doc_template',
    name: 'Document Template',
    document_type: 'TEMPLATE',
    source_url: 'https://example.com',
    text_length: 0,
    word_count: 0,
    page_count: 0,
    language: 'en',
    processing_date: '2024-01-01'
});

// Act nodes represent legal acts and legislation
// Properties: id, name, act_number, year, description, status
CREATE (a:Act:Entity {
    id: 'act_template',
    name: 'Act Template',
    act_number: 'TEMPLATE',
    year: 2024,
    description: 'Template act',
    status: 'active'
});

// Policy nodes represent government orders, policies, guidelines
// Properties: id, name, policy_number, type, effective_date, status, description
CREATE (p:Policy:Entity {
    id: 'policy_template',
    name: 'Policy Template',
    policy_number: 'TEMPLATE',
    type: 'GOVERNMENT_ORDER',
    effective_date: '2024-01-01',
    status: 'active',
    description: 'Template policy'
});

// Rule nodes represent specific rules and regulations
// Properties: id, name, rule_number, description, applicable_to, conditions
CREATE (r:Rule:Entity {
    id: 'rule_template',
    name: 'Rule Template',
    rule_number: 'TEMPLATE',
    description: 'Template rule',
    applicable_to: 'ALL_SCHOOLS',
    conditions: 'Template conditions'
});

// Organization nodes represent institutions, departments, agencies
// Properties: id, name, type, jurisdiction, contact_info, website
CREATE (o:Organization:Entity {
    id: 'org_template',
    name: 'Organization Template',
    type: 'DEPARTMENT',
    jurisdiction: 'ANDHRA_PRADESH',
    contact_info: 'Template contact',
    website: 'https://example.com'
});

// Person nodes represent individuals mentioned in documents
// Properties: id, name, title, role, organization, contact_info
CREATE (per:Person:Entity {
    id: 'person_template',
    name: 'Person Template',
    title: 'Template Title',
    role: 'Template Role',
    organization: 'Template Org',
    contact_info: 'Template contact'
});

// Location nodes represent geographical entities
// Properties: id, name, type, state, district, coordinates
CREATE (loc:Location:Entity {
    id: 'loc_template',
    name: 'Location Template',
    type: 'DISTRICT',
    state: 'ANDHRA_PRADESH',
    district: 'Template District',
    coordinates: '0.0,0.0'
});

// School nodes represent educational institutions
// Properties: id, name, type, level, location, enrollment, staff_count
CREATE (sch:School:Entity {
    id: 'school_template',
    name: 'School Template',
    type: 'GOVERNMENT',
    level: 'PRIMARY',
    location: 'Template Location',
    enrollment: 0,
    staff_count: 0
});

// Curriculum nodes represent educational curricula and syllabi
// Properties: id, name, subject, grade_level, description, effective_date
CREATE (cur:Curriculum:Entity {
    id: 'curriculum_template',
    name: 'Curriculum Template',
    subject: 'Template Subject',
    grade_level: 'Template Grade',
    description: 'Template description',
    effective_date: '2024-01-01'
});

// Assessment nodes represent evaluation methods and standards
// Properties: id, name, type, description, applicable_grades, frequency
CREATE (ass:Assessment:Entity {
    id: 'assessment_template',
    name: 'Assessment Template',
    type: 'FORMATIVE',
    description: 'Template assessment',
    applicable_grades: 'ALL',
    frequency: 'MONTHLY'
});

// Judgment nodes represent court decisions and legal precedents
// Properties: id, name, case_number, court, date, summary, impact
CREATE (j:Judgment:Entity {
    id: 'judgment_template',
    name: 'Judgment Template',
    case_number: 'TEMPLATE',
    court: 'Template Court',
    date: '2024-01-01',
    summary: 'Template summary',
    impact: 'Template impact'
});

// Define relationship types and their properties

// Document relationships
// AMENDS: Document amends another document
CREATE (d1:Document)-[:AMENDS {
    id: 'rel_template_1',
    type: 'AMENDS',
    confidence: 1.0,
    context: 'Template context',
    effective_date: '2024-01-01'
}]->(d2:Document);

// IMPLEMENTS: Document implements a policy or act
CREATE (d1:Document)-[:IMPLEMENTS {
    id: 'rel_template_2',
    type: 'IMPLEMENTS',
    confidence: 1.0,
    context: 'Template context',
    implementation_date: '2024-01-01'
}]->(p1:Policy);

// CONTAINS: Document contains entities
CREATE (d1:Document)-[:CONTAINS {
    id: 'rel_template_3',
    type: 'CONTAINS',
    confidence: 1.0,
    context: 'Template context',
    position: 0
}]->(e1:Entity);

// Policy relationships
// GOVERNED_BY: Policy is governed by an act
CREATE (p1:Policy)-[:GOVERNED_BY {
    id: 'rel_template_4',
    type: 'GOVERNED_BY',
    confidence: 1.0,
    context: 'Template context',
    effective_date: '2024-01-01'
}]->(a1:Act);

// APPLIES_TO: Policy applies to specific entities
CREATE (p1:Policy)-[:APPLIES_TO {
    id: 'rel_template_5',
    type: 'APPLIES_TO',
    confidence: 1.0,
    context: 'Template context',
    conditions: 'Template conditions'
}]->(sch1:School);

// SUPERSEDES: Policy supersedes another policy
CREATE (p1:Policy)-[:SUPERSEDES {
    id: 'rel_template_6',
    type: 'SUPERSEDES',
    confidence: 1.0,
    context: 'Template context',
    supersession_date: '2024-01-01'
}]->(p2:Policy);

// Organization relationships
// ADMINISTERS: Organization administers policies or programs
CREATE (o1:Organization)-[:ADMINISTERS {
    id: 'rel_template_7',
    type: 'ADMINISTERS',
    confidence: 1.0,
    context: 'Template context',
    authority_level: 'STATE'
}]->(p1:Policy);

// OVERSEES: Organization oversees schools or programs
CREATE (o1:Organization)-[:OVERSEES {
    id: 'rel_template_8',
    type: 'OVERSEES',
    confidence: 1.0,
    context: 'Template context',
    oversight_level: 'DISTRICT'
}]->(sch1:School);

// School relationships
// FOLLOWS: School follows curriculum or policy
CREATE (sch1:School)-[:FOLLOWS {
    id: 'rel_template_9',
    type: 'FOLLOWS',
    confidence: 1.0,
    context: 'Template context',
    compliance_level: 'FULL'
}]->(cur1:Curriculum);

// LOCATED_IN: School is located in a specific location
CREATE (sch1:School)-[:LOCATED_IN {
    id: 'rel_template_10',
    type: 'LOCATED_IN',
    confidence: 1.0,
    context: 'Template context',
    coordinates: '0.0,0.0'
}]->(loc1:Location);

// Curriculum relationships
// PART_OF: Curriculum is part of a larger framework
CREATE (cur1:Curriculum)-[:PART_OF {
    id: 'rel_template_11',
    type: 'PART_OF',
    confidence: 1.0,
    context: 'Template context',
    framework_version: '2024'
}]->(cur2:Curriculum);

// ASSESSED_BY: Curriculum is assessed by specific methods
CREATE (cur1:Curriculum)-[:ASSESSED_BY {
    id: 'rel_template_12',
    type: 'ASSESSED_BY',
    confidence: 1.0,
    context: 'Template context',
    assessment_frequency: 'MONTHLY'
}]->(ass1:Assessment);

// Person relationships
// WORKS_FOR: Person works for an organization
CREATE (per1:Person)-[:WORKS_FOR {
    id: 'rel_template_13',
    type: 'WORKS_FOR',
    confidence: 1.0,
    context: 'Template context',
    position: 'Template Position',
    start_date: '2024-01-01'
}]->(o1:Organization);

// AUTHORED: Person authored a document
CREATE (per1:Person)-[:AUTHORED {
    id: 'rel_template_14',
    type: 'AUTHORED',
    confidence: 1.0,
    context: 'Template context',
    role: 'PRIMARY_AUTHOR',
    date: '2024-01-01'
}]->(d1:Document);

// Judgment relationships
// INTERPRETS: Judgment interprets an act or policy
CREATE (j1:Judgment)-[:INTERPRETS {
    id: 'rel_template_15',
    type: 'INTERPRETS',
    confidence: 1.0,
    context: 'Template context',
    interpretation_type: 'BINDING',
    date: '2024-01-01'
}]->(a1:Act);

// AFFECTS: Judgment affects policies or practices
CREATE (j1:Judgment)-[:AFFECTS {
    id: 'rel_template_16',
    type: 'AFFECTS',
    confidence: 1.0,
    context: 'Template context',
    impact_level: 'HIGH',
    effective_date: '2024-01-01'
}]->(p1:Policy);

// Clean up template nodes (they were just for schema definition)
MATCH (n) WHERE n.id CONTAINS 'template' DELETE n;

// Create some sample data to demonstrate the schema
CREATE (ap:Location:Entity {
    id: 'ap_state',
    name: 'Andhra Pradesh',
    type: 'STATE',
    state: 'ANDHRA_PRADESH',
    district: 'ALL',
    coordinates: '15.9129,79.7400'
});

CREATE (mhrd:Organization:Entity {
    id: 'mhrd',
    name: 'Ministry of Human Resource Development',
    type: 'MINISTRY',
    jurisdiction: 'INDIA',
    contact_info: 'New Delhi',
    website: 'https://mhrd.gov.in'
});

CREATE (scert:Organization:Entity {
    id: 'scert_ap',
    name: 'State Council of Educational Research and Training',
    type: 'COUNCIL',
    jurisdiction: 'ANDHRA_PRADESH',
    contact_info: 'Hyderabad',
    website: 'https://apscert.gov.in'
});

CREATE (rte_act:Act:Entity {
    id: 'rte_act_2009',
    name: 'Right to Education Act',
    act_number: 'RTE-2009',
    year: 2009,
    description: 'Fundamental right to free and compulsory education',
    status: 'active'
});

CREATE (nep_policy:Policy:Entity {
    id: 'nep_2020',
    name: 'National Education Policy',
    policy_number: 'NEP-2020',
    type: 'NATIONAL_POLICY',
    effective_date: '2020-07-29',
    status: 'active',
    description: 'Comprehensive education policy framework'
});

// Create relationships between sample entities
CREATE (nep_policy)-[:GOVERNED_BY {
    id: 'nep_rte_relation',
    type: 'GOVERNED_BY',
    confidence: 1.0,
    context: 'NEP 2020 is governed by RTE Act 2009',
    effective_date: '2020-07-29'
}]->(rte_act);

CREATE (scert)-[:ADMINISTERS {
    id: 'scert_nep_relation',
    type: 'ADMINISTERS',
    confidence: 1.0,
    context: 'SCERT administers NEP implementation',
    authority_level: 'STATE'
}]->(nep_policy);

CREATE (scert)-[:LOCATED_IN {
    id: 'scert_location_relation',
    type: 'LOCATED_IN',
    confidence: 1.0,
    context: 'SCERT is located in Andhra Pradesh',
    coordinates: '17.3850,78.4867'
}]->(ap);
