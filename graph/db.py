from neo4j import GraphDatabase

neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "twoje_haslo"))


def ingest_to_neo4j(relations):
    """Save extracted relations to Neo4j."""
    query = """
    MERGE (s:Entity {name: $source, type: $source_type})
    MERGE (t:Entity {name: $target, type: $target_type})
    MERGE (s)-[r:RELATION {type: $relation}]->(t)
    """
    with neo4j_driver.session() as session:
        for rel in relations:
            session.run(
                query,
                source=rel["source"],
                source_type=rel["source_type"],
                target=rel["target"],
                target_type=rel["target_type"],
                relation=rel["relation"],
            )


def search_graph(query: str):
    """Graph search within Neo4j."""
    cypher_query = """
    MATCH (s)-[r]->(t)
    RETURN s.name, type(r), t.name LIMIT 5
    """
    context = []
    with neo4j_driver.session() as session:
        result = session.run(cypher_query)
        for record in result:
            context.append(f"{record['s.name']} {record['type(r)']} {record['t.name']}")
    return context
