# Hybrid RAG for Cyber Threat Intelligence (CTI)

A hybrid Retrieval-Augmented Generation (RAG) system combining the semantic depth of Vector Search (Qdrant) with the relational precision of Knowledge Graphs (Neo4j). The system uses a local LLM (Ollama / Llama 3) to route queries and generate answers securely without data leaving your machine.

## Prerequisites

- **Docker & Docker Compose** (for running Qdrant and Neo4j)
- **Python 3.8+**
- **Ollama** (for local LLM execution)

## 1. Installation & Setup

### Install Python Dependencies
Create a virtual environment (recommended) and install the required packages using the provided `requirements.txt`:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Set Up Local LLM (Ollama)
1. Download and install [Ollama](https://ollama.com/).
2. Open a terminal and pull/run the Llama 3 model:
```bash
ollama run llama3
```
*(Keep the Ollama service running in the background).*

### Optional: Gemma4 setup for dataset generation

If Windows: Download Jan and within it gemma-4-E4B-it-Q8_0 model.
Else: :). 

## 2. Running the Infrastructure

The project uses Docker Compose to spin up the required databases. The `docker-compose.yml` includes Neo4j (pre-configured with APOC and Graph Data Science plugins) and Qdrant.

Start the databases in detached mode:
```bash
docker compose up -d
```

You can verify the services are running at:
* **Neo4j Browser:** http://localhost:7474 (Login: `neo4j` / Password: `twoje_haslo`)
* **Qdrant Dashboard:** http://localhost:6333/dashboard

## 3. Running the Application

Once the infrastructure and Ollama are fully initialized, you can start the main orchestrator script:

```bash
python main.py
```

**What the script does:**
1. Connects to local Qdrant and Neo4j instances.
2. Ingests sample CTI data (Threat Actors, IP Addresses, Vulnerabilities) into the Neo4j graph.
3. Prompts the user for a query (e.g., *"Z jakimi IP komunikuje sie APT29?"*).
4. Uses the local Llama 3 model to classify and route the query (`GRAPH`, `VECTOR`, or `HYBRID`).
5. Retrieves the relevant context and generates a factual, hallucination-free response.

## Shutting Down
To stop the database containers (your data will be preserved in Docker volumes):

```bash
docker compose down
```

## 4. Ground-truth dataset generation

Ragas anlong with gemma-4-E4B-it-Q8_0 were utilized to generate ground-truth dataset.