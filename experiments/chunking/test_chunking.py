import os
import sys
import pandas as pd
import glob
from datasets import Dataset
from pathlib import Path
from typing import List

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from src.rags import GraphRAG, VectorRAG
from src.utils.config import GraphConfig, VectorConfig, get_config
from src.utils.query_routing import route_query

# Constants
CSV_PATH = "experiments/dataset/cti_ground_truth_local.csv"
RAW_DATA_DIR = "raw_data"
STRATEGIES = ["sliding_window", "fixed", "sentence", "semantic"]

config = get_config()

def generate_answer(llm, query: str, vector_context: List[str], graph_context: List[str]) -> str:
    """Generates an answer based on combined vector and graph context."""
    context_str = "\n--- Vector context (Reports) ---\n" + "\n".join(vector_context)
    context_str += "\n\n--- Graph context (Relations) ---\n" + "\n".join(graph_context)

    prompt = f"""You are a Cyberthreat Intelligence (CTI) analyst. Answer the question based EXCLUSIVELY on the provided context.
    If the context does not contain the answer, say that you do not know.

    Context:
    {context_str}

    Question: {query}
    Answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def main():
    # 1. Initialize models and RAGs
    print("Initializing LLM and RAG components...")
    llm = ChatOpenAI(
        base_url=config.llm.api_url,
        api_key="not-needed",
        model=config.llm.model_name,
        temperature=0.0,
        default_headers={"Host": "localhost"}
    )

    wrapped_llm = LangchainLLMWrapper(llm)
    local_langchain_embeddings = HuggingFaceEmbeddings(model_name=config.embedding.model)
    wrapped_embeddings = LangchainEmbeddingsWrapper(local_langchain_embeddings)

    vector_config = VectorConfig.load()
    graph_config = GraphConfig.load()

    vector_rag = VectorRAG(vector_config)
    graph_rag = GraphRAG(graph_config)

    # 2. Load ground truth dataset
    print(f"Loading ground truth dataset from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    q_col = 'user_input' if 'user_input' in df.columns else 'question'
    gt_col = 'reference' if 'reference' in df.columns else 'ground_truth'

    # 3. Get raw data files
    raw_data_path = os.path.abspath(RAW_DATA_DIR)
    pdf_files = sorted(glob.glob(os.path.join(raw_data_path, "*.pdf")))

    if not pdf_files:
        print(f"No PDF files found in {raw_data_path}")
        return

    # 4. Iterate over chunking strategies
    for strategy in STRATEGIES:
        collection_name = f"cti_reports_{strategy}"
        print(f"\n{'='*60}")
        print(f"Testing Strategy: {strategy.upper()}")
        print(f"{'='*60}")

        # Ingest into Vector store
        print(f"Ingesting into Vector collection: {collection_name}...")
        vector_rag.ingest(pdf_files, collection_name=collection_name, strategy=strategy)

        # Ingest into Graph store (clearing existing data first for strategy independence)
        print(f"Clearing and Ingesting Graph Data for {strategy}...")
        with graph_rag.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        graph_rag.ingest(pdf_files, strategy=strategy)

        data_dict = {
            "user_input": [],
            "reference": [],
            "response": [],
            "retrieved_contexts": []
        }

        print("Generating Hybrid RAG answers for evaluation...")
        for idx, row in df.iterrows():
            question = row[q_col]
            ground_truth = row[gt_col]

            try:
                # Use query routing logic
                route_decision = route_query(question, llm=llm)

                vector_data: List[str] = []
                graph_data: List[str] = []

                if route_decision in ["VECTOR", "HYBRID"]:
                    vector_data = vector_rag.search(question, collection_name=collection_name)

                if route_decision in ["GRAPH", "HYBRID"]:
                    graph_data = graph_rag.search(question)

                answer = generate_answer(llm, question, vector_data, graph_data)

                data_dict["user_input"].append(question)
                data_dict["reference"].append(ground_truth)
                data_dict["response"].append(answer)
                # Combine contexts for RAGAS evaluation
                data_dict["retrieved_contexts"].append(vector_data + graph_data)

                print(f"  [{idx+1}/{len(df)}] Route: {route_decision}")
            except Exception as e:
                print(f"  [ERROR] Question {idx}: {e}")
                continue

        # 5. RAGAS Evaluation
        eval_dataset = Dataset.from_dict(data_dict)
        print(f"Evaluating {strategy} results with Ragas metrics...")

        safe_config = RunConfig(max_workers=1, timeout=600)
        metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ]

        score = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_embeddings,
            run_config=safe_config
        )

        # Save results to CSV
        eval_df = score.to_pandas()
        output_file = f"experiments/chunking/results/eval_results_{strategy}.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        eval_df.to_csv(output_file, index=False)

        print(f"Results saved to {output_file}")
        print(f"\nScores for {strategy}:")
        print(score)

    graph_rag.close()
    print("\nRefactored evaluation completed successfully.")

if __name__ == "__main__":
    main()
