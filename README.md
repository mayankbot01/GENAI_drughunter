# ToxiSense: Multi-Modal Toxicity Prediction Agent 🧬🧪

ToxiSense is a revolutionary GenAI-powered toxicity prediction platform designed to solve the Phase 2 clinical trial bottleneck: late-stage safety failures. By fusing molecular graphs, chemical language models, and protein embeddings with RAG over FDA adverse event reports, ToxiSense provides both highly accurate predictions and deep biological explanations.

## 🚀 Key Features

- **Multimodal Fusion**: Combines GNNs (AttentiveFP) for molecular graphs + ChemBERTa-2 for SMILES strings + ESM-2 for protein context.
- **RAG-Grounded Predictions**: Pulls real-world evidence from FDA FAERS and SIDER databases to validate toxicity alerts.
- **Agentic Explainability**: A LangGraph-powered agent uses LLMs to explain *why* a molecule is toxic, identifying specific SMARTS patterns and biological mechanisms.
- **Toxicity Neutralization**: Suggests structural modifications to "fix" toxic molecules while maintaining efficacy.
- **Drug Discovery MCP**: Exposed as a Model Context Protocol (MCP) server for seamless integration into larger drug discovery agent workflows.

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, PyTorch Geometric, HuggingFace Transformers, ESM-2 (Meta AI).
- **Chemical Informatics**: RDKit, DeepChem.
- **GenAI & Agents**: LangChain, LangGraph, LlamaIndex, OpenAI GPT-4o / Claude 3.5 Sonnet.
- **RAG & Vector Search**: ChromaDB / FAISS, pgvector.
- **LLMOps & Monitoring**: MLflow, Weights & Biases, Arize Phoenix.
- **Deployment**: FastAPI, Docker, Kubernetes, Streamlit.

## 📁 Project Structure

```text
├── app/                  # Streamlit frontend dashboard
├── configs/              # Hyperparameters and model configs
├── data/                 # Data storage (Tox21, ClinTox, FAERS)
├── notebooks/            # EDA and training walkthroughs
├── src/
│   ├── agents/           # LangGraph agents for toxicity analysis
│   ├── api/              # FastAPI serving endpoints
│   ├── data/             # Data loaders and preprocessing (RDKit)
│   ├── explainability/   # SMARTS explainer and LLM mechanistic reasoning
│   ├── models/           # GNN, ChemBERTa, and Fusion model architectures
│   ├── pipeline/         # Training and evaluation orchestration
│   └── rag/              # RAG over FAERS and SIDER
├── tests/                # Unit and integration tests
├── Dockerfile            # Containerization
├── requirements.txt      # Dependency management
└── README.md             # Project documentation
```

## 📊 Performance Benchmarks

Targeting State-of-the-Art (SOTA) on:
- **Tox21 Challenge**: >0.85 AUC across all 12 tasks.
- **ClinTox**: >0.92 AUC for clinical toxicity.
- **SIDER**: High-precision side effect mapping.

## 🛠️ Setup & Installation

```bash
git clone https://github.com/mayankbot01/GENAI_drughunter.git
cd GENAI_drughunter
pip install -r requirements.txt
```

## 🧪 Quick Start

```python
from src.agents.toxicity_agent import ToxicityAgent

agent = ToxicityAgent(model_path="models/fusion_v1.pt")
molecule_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
result = agent.analyze(molecule_smiles)

print(f"Toxicity Prediction: {result['prediction']}")
print(f"Reasoning: {result['explanation']}")
```

---
**Maintained by mayankbot01** | GenAI Drug Discovery Expert
