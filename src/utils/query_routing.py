from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


def route_query(query: str, llm=None) -> str:
    """
    LLM-based routing for CTI queries.
    
    Determines whether a query should be routed to vector search, graph search, or both.
    
    Args:
        query: The user's question to route.
        llm: Optional LLM instance. If None, creates a default ChatOllama instance.
        
    Returns:
        One of: "VECTOR", "GRAPH", "HYBRID"
    """
    if llm is None:
        llm = ChatOllama(model="llama3", temperature=0)
    
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
