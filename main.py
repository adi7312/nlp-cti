from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import glob
import os
from vector.db import ingest_pdfs_to_qdrant, search_vector
from graph.db import ingest_to_neo4j, search_graph
from config import EMBEDDING_MODEL

llm = ChatOllama(model="llama3", temperature=0) # firstly set up local ollama model, https://ollama.com/download/linux

# example graph rag data, dummy :/
extracted_relations = [
    {"source": "APT29", "source_type": "ThreatActor", "relation": "USES_VULNERABILITY", "target": "CVE-2021-26855", "target_type": "Vulnerability"},
    {"source": "APT29", "source_type": "ThreatActor", "relation": "COMMUNICATES_WITH", "target": "192.168.1.50", "target_type": "IP_Address"}
]

    

def route_query(query: str) -> str:
    """
    LLM-based routing
    """

    system_prompt = """
    You are a routing system for a CTI (Cyber Threat Intelligence) database.
    Analyze the question and choose ONE path:
    - Return "VECTOR" if the question concerns general descriptions, definitions, behaviors, or methods.
    - Return "GRAPH" if the question concerns explicit connections (e.g., what IPs, what vulnerabilities, who is connected to what).
    - Return "HYBRID" if the question requires both types of information.
    Return ONLY one word from the above.
    """
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    return response.content.strip().upper()


def generate_answer(query: str, vector_context: list, graph_context: list):

    context_str = "\n--- Vector context (Reports) ---\n" + "\n".join(vector_context)
    context_str += "\n\n--- Graph context (Relations) ---\n" + "\n".join(graph_context)
    
    prompt = f"""
    You are a CTI analyst. Answer the question based EXCLUSIVELY on the provided context.
    If the context does not contain the answer, say that you do not know.
    
    Context:
    {context_str}
    
    Question: {query}
    Answer:
    """
    print(prompt)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def main():
    raw_data_path = os.path.join(os.path.dirname(__file__), "raw_data")
    pdf_files = sorted(glob.glob(os.path.join(raw_data_path, "*.pdf")))
    
    if not pdf_files:
        print(f"No PDF files found in {raw_data_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s): {pdf_files}\n")
    
    print("--- Ingesting Graph Data ---")
    ingest_to_neo4j(extracted_relations)
    
    print("\n--- Ingesting Vector Data ---")

    strategies = ["sliding_window", "fixed", "sentence", "semantic"]
    for s in strategies:
        ingest_pdfs_to_qdrant(pdf_files, strategy=s) 
    
    print("\n--- System Ready ---")
    while True:
        user_query = input("\nAsk question (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
            
        route_decision = route_query(user_query)
        print(f"Routing decision: {route_decision}")
        
        vector_data = []
        graph_data = []
        
        if route_decision in ["VECTOR", "HYBRID"]:
            print("Performing vector search...")
            vector_data = search_vector(user_query)
            
        if route_decision in ["GRAPH", "HYBRID"]:
            print("Performing graph search...")
            graph_data = search_graph(user_query)
            
        final_answer = generate_answer(user_query, vector_data, graph_data)
        
        print("\n================ Response ================")
        print(final_answer)
        print("=======================================")
    print("=======================================")

if __name__ == "__main__":
    main()