from neo4j import GraphDatabase
from typing import List, Dict, Any

from .base import BaseRAG
from utils.config import GraphConfig


class GraphRAG(BaseRAG):
    """Graph RAG implementation using Neo4j as the graph database."""

    def __init__(self, config: GraphConfig):
        """Initialize the GraphRAG with Neo4j driver.

        Args:
            config: GraphConfig instance with uri, user, and password.
        """
        self.neo4j_driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))

    def init_storage(self, name: str, **kwargs: Dict[str, Any]) -> None:
        """Initialize Neo4j storage with indexes for better query performance.

        Args:
            name: Database name (unused for Neo4j, kept for interface compatibility).
            **kwargs: Additional parameters (unused).
        """
        # Create indexes for Entity nodes if they don't exist
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        with self.neo4j_driver.session() as session:
            for query in index_queries:
                session.run(query)

    def ingest(self, data: List[Dict[str, Any]], **kwargs: Dict[str, Any]) -> None:
        """Save extracted relations to Neo4j.

        Args:
            data: List of relation dictionaries with keys: source, source_type,
                  target, target_type, relation.
            **kwargs: Additional parameters (unused).
        """
        query = """
        MERGE (s:Entity {name: $source, type: $source_type})
        MERGE (t:Entity {name: $target, type: $target_type})
        MERGE (s)-[r:RELATION {type: $relation}]->(t)
        """
        with self.neo4j_driver.session() as session:
            for rel in data:
                session.run(
                    query,
                    source=rel["source"],
                    source_type=rel["source_type"],
                    target=rel["target"],
                    target_type=rel["target_type"],
                    relation=rel["relation"],
                )

    def search(self, query: str, **kwargs: Dict[str, Any]) -> List[str]:
        """Graph search within Neo4j.

        Args:
            query: Search query (currently unused; runs predefined Cypher).
            **kwargs: Additional parameters including:
                - limit: Maximum number of results (default: 5)

        Returns:
            List of formatted relationship strings.
        """
        limit = kwargs.get("limit", 5)
        cypher_query = f"""
        MATCH (s)-[r]->(t)
        RETURN s.name, type(r), t.name LIMIT {limit}
        """
        context = []
        with self.neo4j_driver.session() as session:
            result = session.run(cypher_query)
            for record in result:
                context.append(f"{record['s.name']} {record['type(r)']} {record['t.name']}")
        return context
