from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional, cast
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import numpy as np

from src.rags.base import BaseRAG
from src.utils.config import GraphConfig
from src.utils.extraction import EntityRelationExtractor
from src.utils.community_detection import LeidenAlgorithm
from src.utils.chunk_strategies import chunk_text


class GraphRAG(BaseRAG):
    """Graph RAG implementation using Neo4j as the graph database.

    This implementation uses BERT-BiLSTM-CRF for entity and relation extraction
    and the Leiden algorithm for community detection.
    """

    def __init__(self, config: GraphConfig, extractor_path: Optional[str] = None):
        """Initialize the GraphRAG with Neo4j driver and EntityRelationExtractor.

        Args:
            config: GraphConfig instance with uri, user, password, and models.
            extractor_path: Optional path to load a pre-trained extractor.
        """
        self.neo4j_driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
        self.extractor = EntityRelationExtractor(bert_model_name=config.bert_model_name)
        self.leiden = LeidenAlgorithm()
        self.lc_embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model)

        if extractor_path and os.path.exists(extractor_path):
            self._load_extractor(extractor_path)

    def init_storage(self, name: str = "neo4j", **kwargs: Dict[str, Any]) -> None:
        """Initialize Neo4j storage with indexes and constraints.

        Args:
            name: Database name (unused for Neo4j, kept for interface compatibility).
            **kwargs: Additional parameters (unused).
        """
        index_queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.community)",
        ]
        with self.neo4j_driver.session() as session:
            for query in index_queries:
                try:
                    session.run(query)
                except Exception as e:
                    print(f"Index/Constraint creation failed: {e}")

    def ingest(self, data: List[str], **kwargs: Dict[str, Any]) -> None:
        """Load PDFs, extract entities/relations using BERT-BiLSTM-CRF, and save to Neo4j.

        Args:
            data: List of PDF file paths to ingest.
            **kwargs: Additional parameters including:
                - strategy: Chunking strategy (default: "sliding_window")
        """
        strategy = cast(str, kwargs.get("strategy", "sliding_window"))

        # Initialize storage (ensure constraints/indexes)
        self.init_storage()

        all_chunks = []
        for path in data:
            print(f"Loading {path}...")
            loader = PyPDFLoader(path)
            docs = loader.load()
            chunks = chunk_text(docs, strategy=strategy, embedding_model=self.lc_embedding_model)
            all_chunks.extend(chunks)

        print(f"Extracting entities and relations from {len(all_chunks)} chunks...")

        all_extractions = []
        for chunk in all_chunks:
            text = chunk.page_content
            extraction = self.extractor.extract(text)
            all_extractions.append(extraction)

        self._save_extractions_batched(all_extractions)

        print("Computing communities...")
        self._compute_communities()

        print("Ingestion complete!")

    def search(self, query: str, **kwargs: Dict[str, Any]) -> List[str]:
        """Perform a graph-aware search.

        This implementation:
        1. Extracts entities from the query.
        2. Finds related entities and their communities.
        3. Retrieves relevant relationships as context.

        Args:
            query: Search query string.
            **kwargs: Additional parameters including:
                - limit: Max context items to retrieve (default: 10).

        Returns:
            List of formatted context strings.
        """
        # Extract entities from query
        query_extraction = self.extractor.extract(query)
        tokens = query_extraction['tokens']
        query_entities = []
        for ent in query_extraction['entities']:
            query_entities.append(" ".join(tokens[ent['start']:ent['end']]))

        if not query_entities:
            return self._fallback_search(kwargs.get("limit", 10))

        limit = kwargs.get("limit", 10)
        context = []

        with self.neo4j_driver.session() as session:
            # Search for relationships involving query entities or their communities
            cypher = """
            MATCH (e:Entity) WHERE e.name IN $names

            // Get direct neighbors
            OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
            WITH e, COLLECT(DISTINCT {name: neighbor.name, type: type(r)})[0..$limit] AS neighbors

            // Get community members
            OPTIONAL MATCH (comm_member:Entity)
            WHERE comm_member.community = e.community AND e.community IS NOT NULL AND comm_member.name <> e.name
            WITH e, neighbors, COLLECT(DISTINCT comm_member.name)[0..$limit] AS comm_members

            RETURN e.name AS entity, neighbors, comm_members
            """
            result = session.run(cypher, names=query_entities, limit=limit)

            for record in result:
                entity_name = record['entity']

                # Add neighbor relationships
                for rel in record['neighbors']:
                    if rel['name']:
                        context.append(f"{entity_name} {rel['type']} {rel['name']}")

                # Add community context
                for member_name in record['comm_members']:
                    context.append(f"Entity '{entity_name}' is in the same community as '{member_name}'")

        return list(set(context))[:limit]

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.neo4j_driver:
            self.neo4j_driver.close()

    # ----------------------------------------------------------------------------------------------------

    def _load_extractor(self, path: str) -> None:
        """Load a pre-trained EntityRelationExtractor from disk.

        Args:
            path: Directory containing the saved extractor models.
        """
        print(f"Loading EntityRelationExtractor from {path}...")
        self.extractor.load(path)

    def _save_extractions_batched(self, extractions: List[Dict[str, Any]]) -> None:
        """Saves a list of extractions to Neo4j using batched UNWIND queries for efficiency."""
        all_entities = []
        all_relations = []

        for ext in extractions:
            tokens = ext['tokens']
            entities = ext['entities']
            relations = ext['relations']

            entity_map = {}
            for i, ent in enumerate(entities):
                name = " ".join(tokens[ent['start']:ent['end']])
                etype = ent['type']
                entity_map[i] = (name, etype)
                all_entities.append({'name': name, 'type': etype})

            for rel in relations:
                head_idx = rel['head']
                tail_idx = rel['tail']
                rtype = rel['type']

                if head_idx in entity_map and tail_idx in entity_map:
                    h_name, _ = entity_map[head_idx]
                    t_name, _ = entity_map[tail_idx]
                    safe_rtype = rtype.upper().replace(" ", "_")
                    all_relations.append({
                        'h_name': h_name,
                        't_name': t_name,
                        'type': safe_rtype
                    })

        with self.neo4j_driver.session() as session:
            # Batch create entities
            session.run(
                "UNWIND $batch AS entity MERGE (e:Entity {name: entity.name}) SET e.type = entity.type",
                batch=all_entities
            )

            # Batch create relations (grouped by type to use dynamic relationship types safely)
            rel_by_type = {}
            for rel in all_relations:
                rtype = rel['type']
                if rtype not in rel_by_type:
                    rel_by_type[rtype] = []
                rel_by_type[rtype].append(rel)

            for rtype, batch in rel_by_type.items():
                session.run(
                    f"""
                    UNWIND $batch AS rel
                    MATCH (h:Entity {{name: rel.h_name}})
                    MATCH (t:Entity {{name: rel.t_name}})
                    MERGE (h)-[r:{rtype}]->(t)
                    """,
                    batch=batch
                )

    def _compute_communities(self) -> None:
        """Compute communities of entities using the Leiden algorithm.

        Args:
            resolution: Resolution parameter for Leiden algorithm.
        """
        print("Fetching graph for community detection...")

        with self.neo4j_driver.session() as session:
            nodes_res = session.run("MATCH (e:Entity) RETURN e.name AS name")
            nodes = [record["name"] for record in nodes_res]

            if not nodes:
                print("No entities found in the graph.")
                return

            node_to_idx = {name: i for i, name in enumerate(nodes)}
            n = len(nodes)

            # Build adjacency list (sparse representation)
            edges_res = session.run(
                "MATCH (e1:Entity)-[r]->(e2:Entity) RETURN e1.name AS s, e2.name AS t, count(r) AS w"
            )

            adj_list = {i: {} for i in range(n)}
            for record in edges_res:
                s_idx = node_to_idx.get(record["s"])
                t_idx = node_to_idx.get(record["t"])
                if s_idx is not None and t_idx is not None:
                    weight = float(record["w"])
                    adj_list[s_idx][t_idx] = adj_list[s_idx].get(t_idx, 0.0) + weight
                    adj_list[t_idx][s_idx] = adj_list[t_idx].get(s_idx, 0.0) + weight

        print(f"Running Leiden algorithm on {n} nodes...")
        labels = self.leiden.fit(adj_list)

        print("Updating communities in Neo4j...")
        # Use UNWIND for batched update
        updates = [{'name': nodes[i], 'comm': int(labels[i])} for i in range(n)]
        with self.neo4j_driver.session() as session:
            session.run(
                "UNWIND $updates AS update MATCH (e:Entity {name: update.name}) SET e.community = update.comm",
                updates=updates
            )
        print(f"Detected {len(np.unique(labels))} communities.")

    def _fallback_search(self, limit: int) -> List[str]:
        """Basic search that returns arbitrary relationships."""
        cypher = f"MATCH (s:Entity)-[r]->(t:Entity) RETURN s.name AS s, type(r) AS type, t.name AS t LIMIT {limit}"
        context = []
        with self.neo4j_driver.session() as session:
            result = session.run(cypher)
            for record in result:
                context.append(f"{record['s']} {record['type']} {record['t']}")
        return context
