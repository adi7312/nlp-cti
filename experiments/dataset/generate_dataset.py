import argparse
import os
import sys
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from pathlib import Path

# Add project root to path so config can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CTI ground truth dataset using RAGas")
    parser.add_argument("--config", type=str, default=None, help="Path to config.toml file (default: config.toml)")
    parser.add_argument("--data-dir", type=str, default="raw_data", help="Directory containing PDF files (default: raw_data)")
    parser.add_argument("--output-csv", type=str, default="cti_ground_truth_local.csv", help="Output CSV file path (default: cti_ground_truth_local.csv)")
    parser.add_argument("--test-size", type=int, default=10, help="Number of test items to generate (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum workers for Ragas (default: 1)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds (default: 600)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config from specified path or default location
    config_path = args.config
    if config_path is not None:
        config_path = Path(config_path)
    config = load_config(config_path)

    # Use command-line arguments or fall back to config.toml values
    EMBEDDING_MODEL = config.get("embedding", {}).get("model", "BAAI/bge-small-en-v1.5")
    LOCAL_API_URL = config.get("llm", {}).get("api_url", "http://localhost:11434/v1")
    LLM_MODEL_NAME = config.get("llm", {}).get("model_name", "llama3")

    print(f"Loading PDFs from directory: {args.data_dir}")
    loader = DirectoryLoader(
        args.data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} total pages.")

    print(f"Initializing {LLM_MODEL_NAME} model via OpenAI compatible API...")

    local_llm = ChatOpenAI(
        base_url=LOCAL_API_URL,
        api_key="not-needed",
        model=LLM_MODEL_NAME,
        temperature=args.temperature,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    print("Initializing HuggingFace Embeddings...")
    local_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    wrapped_llm = LangchainLLMWrapper(local_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

    print("Configuring Ragas Testset Generator...")
    generator = TestsetGenerator(
        llm=wrapped_llm,
        embedding_model=wrapped_embeddings
    )

    run_config = RunConfig(
        max_workers=args.max_workers,
        timeout=args.timeout
    )

    print(f"Building Knowledge Graph and generating {args.test_size} items. This may take a while...")
    try:
        dataset = generator.generate_with_langchain_docs(
            documents,
            testset_size=args.test_size
        )

        # Export to Pandas DataFrame and CSV
        df = dataset.to_pandas()
        df.to_csv(args.output_csv, index=False)
        print(f"Success! Ground truth dataset saved to {args.output_csv}")

        print("\nSample Output:")
        print(df.head(2))

    except Exception as e:
        print(f"An error occurred during generation: {e}")


if __name__ == "__main__":
    main()
