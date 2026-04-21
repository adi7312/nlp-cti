from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage


llm = ChatOllama(model="llama3", temperature=0) # firstly set up local ollama model, https://ollama.com/download/linux
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# docker compose up :)
qdrant_client = QdrantClient(host="localhost", port=6333)
neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "twoje_haslo"))

# example graph rag data, dummy :/
extracted_relations = [
    {"source": "APT29", "source_type": "ThreatActor", "relation": "USES_VULNERABILITY", "target": "CVE-2021-26855", "target_type": "Vulnerability"},
    {"source": "APT29", "source_type": "ThreatActor", "relation": "COMMUNICATES_WITH", "target": "192.168.1.50", "target_type": "IP_Address"}
]

def ingest_to_neo4j(relations):
    """
    Save extracted relations to Neo4j
    """

    query = """
    MERGE (s:Entity {name: $source, type: $source_type})
    MERGE (t:Entity {name: $target, type: $target_type})
    MERGE (s)-[r:RELATION {type: $relation}]->(t)
    """
    with neo4j_driver.session() as session:
        for rel in relations:
            session.run(query, 
                        source=rel["source"], source_type=rel["source_type"],
                        target=rel["target"], target_type=rel["target_type"],
                        relation=rel["relation"])


# --- 3. ROUTING ZAPYTAŃ (LLM-BASED ROUTING) ---
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

def search_vector(query: str):
    """
    Semantic search within Qdrant.
    """

    query_vector = embedding_model.encode(query).tolist()
    results = qdrant_client.search(collection_name="cti_reports", query_vector=query_vector, limit=2)
    return [hit.payload['text'] for hit in results]

def search_graph(query: str):
    """
    Graph search within neo4j
    """

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
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def main():

    ingest_to_neo4j(extracted_relations)
    user_query = input("Ask question: ")
    
    route_decision = route_query(user_query)
    print(f"Routing decision: {route_decision}")
    
    vector_data = []
    graph_data = []
    
    if route_decision in ["VECTOR", "HYBRID"]:
        print("Performing vector search")
        vector_data = search_vector(user_query)
        
    if route_decision in ["GRAPH", "HYBRID"]:
        print("Performing graph search")
        graph_data = search_graph(user_query)
        
    final_answer = generate_answer(user_query, vector_data, graph_data)
    
    print("\n================ Response ================")
    print(final_answer)
    print("=======================================")

if __name__ == "__main__":
    main()