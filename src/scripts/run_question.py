import argparse
from pathlib import Path
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import glob
import os

from rags import GraphRAG, VectorRAG
from utils.config import GraphConfig, VectorConfig
from utils.query_routing import route_query


STRATEGIES = ["sliding_window", "fixed", "sentence", "semantic"]

# Example graph RAG data for demonstration
EXTRACTED_RELATIONS = [
    {"source": "APT29", "source_type": "ThreatActor", "relation": "USES_VULNERABILITY", "target": "CVE-2021-26855", "target_type": "Vulnerability"},
    {"source": "APT29", "source_type": "ThreatActor", "relation": "COMMUNICATES_WITH", "target": "192.168.1.50", "target_type": "IP_Address"}
]


def parse_args():
    parser = argparse.ArgumentParser(description="CTI Question Answering System with RAG")
    parser.add_argument("--config", type=str, default=None, help="Path to config.toml file (default: config.toml)")
    parser.add_argument("--model", type=str, default="llama3", help="LLM model name (default: llama3)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument("--data-dir", type=str, default="raw_data", help="Directory containing PDF files (default: raw_data)")
    parser.add_argument("--vector-collection", type=str, default="cti_reports", help="Vector collection name (default: cti_reports)")
    parser.add_argument("--graph-collection", type=str, default="cti_graph", help="Graph collection name (default: cti_graph)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion and go straight to querying")
    return parser.parse_args()


def generate_answer(llm, query: str, vector_context: List[str], graph_context: List[str]) -> str:
    context_str = "\n--- Vector context (Reports) ---\n" + "\n".join(vector_context)
    context_str += "\n\n--- Graph context (Relations) ---\n" + "\n".join(graph_context)

    prompt = f"""You are a CTI analyst. Answer the question based EXCLUSIVELY on the provided context.
    If the context does not contain the answer, say that you do not know.

    Context:
    {context_str}

    Question: {query}
    Answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def main() -> None:
    args = parse_args()

    # Initialize LLM with parsed arguments
    llm = ChatOllama(model=args.model, temperature=args.temperature)

    # Load configs from specified path or default location
    config_path = args.config
    if config_path is not None:
        config_path = Path(config_path)
    vector_config = VectorConfig.load(config_path)
    graph_config = GraphConfig.load(config_path)

    vector_rag = VectorRAG(vector_config)
    graph_rag = GraphRAG(graph_config)

    # Initialize storage
    graph_rag.init_storage(args.graph_collection)

    raw_data_path = os.path.join(os.path.dirname(__file__), args.data_dir)
    pdf_files = sorted(glob.glob(os.path.join(raw_data_path, "*.pdf")))

    if not pdf_files:
        print(f"No PDF files found in {raw_data_path}")
        return

    print(f"Found {len(pdf_files)} PDF file(s): {pdf_files}\n")

    if not args.skip_ingest:
        print("--- Ingesting Graph Data ---")
        graph_rag.ingest(EXTRACTED_RELATIONS)

        print("\n--- Ingesting Vector Data ---")
        for strategy in STRATEGIES:
            vector_rag.ingest(pdf_files, collection_name=args.vector_collection, strategy=strategy)

        print("\n--- System Ready ---")
    else:
        print("--- Skipping ingestion (as requested) ---")
        print("--- System Ready ---")

    while True:
        user_query = input("\nAsk question (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        route_decision = route_query(user_query, llm=llm)
        print(f"Routing decision: {route_decision}")

        vector_data: List[str] = []
        graph_data: List[str] = []

        if route_decision in ["VECTOR", "HYBRID"]:
            print("Performing vector search...")
            vector_data = vector_rag.search(user_query, collection_name=args.vector_collection)

        if route_decision in ["GRAPH", "HYBRID"]:
            print("Performing graph search...")
            graph_data = graph_rag.search(user_query)

        final_answer = generate_answer(llm, user_query, vector_data, graph_data)

        print("\n================ Response ================")
        print(final_answer)
        print("=======================================")
    print("Goodbye!")


if __name__ == "__main__":
    main()
