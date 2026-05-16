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
### Chat-mode

Use `run_question.py` with command-line arguments:

```bash
python src/scripts/run_question.py
```

**Available arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.toml` | Path to config.toml file |
| `--model` | `llama3` | LLM model name for Ollama |
| `--temperature` | `0.0` | LLM temperature setting |
| `--data-dir` | `raw_data` | Directory containing PDF files to ingest |
| `--vector-collection` | `cti_reports` | Qdrant collection name for vector data |
| `--graph-collection` | `cti_graph` | Neo4j collection name for graph data |
| `--skip-ingest` | - | Skip data ingestion and go straight to querying |

**Examples:**
```bash
# Use a different model with higher temperature
python src/scripts/run_question.py --model llama3 --temperature 0.7

# Skip ingestion (use existing data)
python src/scripts/run_question.py --skip-ingest

# Custom data directory and collection names
python src/scripts/run_question.py --data-dir my_pdfs --vector-collection my_vectors --graph-collection my_graph
```

**What the script does:**
1. Loads configuration and initializes GraphRAG and VectorRAG instances.
2. Ingests PDF files from the specified directory using multiple chunking strategies.
3. Ingests sample graph relations into Neo4j.
4. Enters an interactive loop where you can ask questions.
5. Routes each query, retrieves relevant context, and generates answers.

## Shutting Down
To stop the database containers (your data will be preserved in Docker volumes):

```bash
docker compose down
```

## 4. Ground-truth dataset generation

To generate a ground-truth dataset for evaluation, use the `generate_dataset.py` script:

```bash
python experiments/dataset/generate_dataset.py
```

**Available arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.toml` | Path to config.toml file |
| `--data-dir` | `raw_data` | Directory containing PDF files |
| `--output-csv` | `cti_ground_truth_local.csv` | Output CSV file path |
| `--test-size` | `10` | Number of test items to generate |
| `--temperature` | `0.0` | LLM temperature |
| `--max-workers` | `1` | Maximum workers for Ragas |
| `--timeout` | `600` | Timeout in seconds |

**Examples:**
```bash
# Generate dataset with default settings
python experiments/dataset/generate_dataset.py

# Generate 20 items with custom output
python experiments/dataset/generate_dataset.py --test-size 20 --output-csv my_dataset.csv

# Use custom config file and model
python experiments/dataset/generate_dataset.py --config custom_config.toml --llm-model gemma-4-E4B-it-Q8_0

# Override embedding model
python experiments/dataset/generate_dataset.py --embedding-model BAAI/bge-small-en-v1.5
```

Ragas along with gemma-4-E4B-it-Q8_0 were utilized to generate ground-truth dataset.
