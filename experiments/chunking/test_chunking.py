import os
import sys
import pandas as pd
from datasets import Dataset
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
from config import EMBEDDING_MODEL, LOCAL_API_URL, LLM_MODEL_NAME


CSV_PATH = "test/dataset/cti_ground_truth_local.csv"
STRATEGIES = ["sliding_window", "fixed", "sentence", "semantic"] # Update these to match your exact Qdrant suffixes

MODEL_NAME = LLM_MODEL_NAME

qdrant_client = QdrantClient("http://localhost:6333") 
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
local_llm = ChatOpenAI(
    base_url=LOCAL_API_URL,
    api_key="not-needed", 
    model=MODEL_NAME, 
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}},
    default_headers={"Host": "localhost"} 
)

wrapped_llm = LangchainLLMWrapper(local_llm)
local_langchain_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
wrapped_embeddings = LangchainEmbeddingsWrapper(local_langchain_embeddings)


rag_prompt = ChatPromptTemplate.from_template("""
You are a Cyberthreat Intelligence Analyst. Answer the question based ONLY on the following context. 
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:
""")

rag_chain = rag_prompt | local_llm


def answer_question_with_rag(question: str, collection_name: str, top_k: int = 3):
    """Embeds the question, searches Qdrant, and generates an answer."""
    query_vector = embedding_model.encode(question).tolist()
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    
    retrieved_contexts = [hit.payload.get("text", "") for hit in search_results]
    combined_context = "\n\n---\n\n".join(retrieved_contexts)
    
    response = rag_chain.invoke({"context": combined_context, "question": question})
    
    return response.content, retrieved_contexts


def main():
    print(f"Loading ground truth dataset from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    q_col = 'user_input' if 'user_input' in df.columns else 'question'
    gt_col = 'reference' if 'reference' in df.columns else 'ground_truth'
    
    evaluation_results = {}

    for strategy in STRATEGIES:
        collection_name = f"cti_reports_{strategy}"
        print(f"\n=============================================")
        print(f"Testing Strategy: {strategy.upper()} (Collection: {collection_name})")
        print(f"=============================================")
        
        data_dict = {
            "user_input": [],
            "reference": [],
            "response": [],
            "retrieved_contexts": []
        }
        
        print("Generating RAG answers...")
        for idx, row in df.iterrows():
            question = row[q_col]
            ground_truth = row[gt_col]
            
            try:
                answer, contexts = answer_question_with_rag(question, collection_name)
                
                data_dict["user_input"].append(question)
                data_dict["reference"].append(ground_truth)
                data_dict["response"].append(answer)
                data_dict["retrieved_contexts"].append(contexts)
            except Exception as e:
                print(f"Error processing question {idx}: {e}")
                continue
                
        eval_dataset = Dataset.from_dict(data_dict)
        
        print("Evaluating RAG performance with Ragas (this will take time)...")
        safe_config = RunConfig(max_workers=1, timeout=600) 
        
        metrics = [
            context_precision, # Did it retrieve the *right* chunks?
            context_recall,    # Did it retrieve *all* the necessary information?
            faithfulness,      # Is the answer hallucinated or based on the chunks?
            answer_relevancy   # Does the answer actually address the question?
        ]
        
        score = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_embeddings,
            run_config=safe_config
        )
        
        eval_df = score.to_pandas()
        output_file = f"eval_results_{strategy}.csv"
        eval_df.to_csv(output_file, index=False)
        print(f"Saved evaluation for {strategy} to {output_file}")
        
        print(f"\n--- Score Summary for {strategy} ---")
        print(score)
        
if __name__ == "__main__":
    main()