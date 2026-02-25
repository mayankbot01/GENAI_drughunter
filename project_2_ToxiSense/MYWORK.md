# PROJECT 2 — ToxiSense: Multi-Modal Toxicity Prediction Agent 🧬🧪

## Bottleneck Solved
Late-stage safety failures in Phase 2 clinical trials costing $100M+ per failed candidate.

## Vision
ToxiSense is a revolutionary GenAI-powered toxicity prediction platform designed to solve the Phase 2 clinical trial bottleneck. By fusing molecular graphs, chemical language models, and protein embeddings with RAG over FDA adverse event reports, ToxiSense provides both highly accurate predictions and deep biological explanations.

## Tech Stack
- **Deep Learning**: PyTorch, PyTorch Geometric, HuggingFace Transformers, ESM-2 (Meta AI).
- **Chemical Informatics**: RDKit, DeepChem.
- **GenAI & Agents**: LangChain, LangGraph, LlamaIndex, OpenAI GPT-4o / Claude 3.5 Sonnet.
- **RAG & Vector Search**: ChromaDB / FAISS, pgvector.

## Project Structure
- `src/agents/`: LangGraph agents for toxicity analysis.
- `src/rag/`: RAG over FAERS and SIDER.
- `src/models/`: GNN, ChemBERTa, and Fusion model architectures.
- `app/`: Streamlit frontend dashboard.

## Key Features
- **Multimodal Fusion**: Combines GNNs (AttentiveFP) + ChemBERTa-2 + ESM-2.
- **RAG-Grounded Predictions**: Evidence from FDA FAERS and SIDER.
- **Agentic Explainability**: LangGraph-powered mechanistic reasoning.
- **Toxicity Neutralization**: Structural modification suggestions.
- **Drug Discovery MCP**: MCP server integration.

---
Maintained by mayankbot01 | GENAI_drughunter Series — Project 2 of 10
