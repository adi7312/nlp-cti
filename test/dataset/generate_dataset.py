import os
import sys
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Add project root to path so config can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from config import EMBEDDING_MODEL, LOCAL_API_URL, LLM_MODEL_NAME


RAW_DATA_DIR = "raw_data" 
OUTPUT_CSV_PATH = "cti_ground_truth_local.csv"
TEST_SIZE = 10 


print(f"Loading PDFs from directory: {RAW_DATA_DIR}")
loader = DirectoryLoader(
    RAW_DATA_DIR, 
    glob="**/*.pdf", 
    loader_cls=PyPDFLoader,
    show_progress=True 
)
documents = loader.load()
print(f"Successfully loaded {len(documents)} total pages.")

print(f"Initializing Local Gemma model via OpenAI compatible API...")

local_llm = ChatOpenAI(
    base_url=LOCAL_API_URL,
    api_key="not-needed", 
    model=LLM_MODEL_NAME, 
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

print("Initializing Local HuggingFace Embeddings...")
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

safe_config = RunConfig(
    max_workers=1, # Forces Ragas to wait for Jan to finish before sending the next prompt
    timeout=600    # Gives your local model plenty of time to read dense PDFs
)

print(f"Building Knowledge Graph and generating {TEST_SIZE} items. This may take a while locally...")
try:
    dataset = generator.generate_with_langchain_docs(
        documents, 
        testset_size=TEST_SIZE
    )
    
    # 7. Export to Pandas DataFrame and CSV
    df = dataset.to_pandas()
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Success! Ground truth dataset saved to {OUTPUT_CSV_PATH}")
    
    print("\nSample Output:")
    print(df.head(2))

except Exception as e:
    print(f"An error occurred during generation: {e}")