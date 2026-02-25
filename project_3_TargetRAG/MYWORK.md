# PROJECT 3 — TargetRAG: Target Retrieval-Augmented Generation Agent 🧬🔍

## Bottleneck Solved
Target identification and validation: 90% of drugs fail in clinical trials because they were tested against the wrong biological target or the target was not druggable.

## Vision
TargetRAG automates the identification and validation of high-potential drug targets by combining cross-modal data retrieval (literature, genomic, and structural databases) with agentic reasoning to score targets for druggability, novelty, and safety.

## Tech Stack
- **Knowledge Retrieval**: LlamaIndex, ChromaDB, LangChain.
- **Data Sources**: PubMed (via Entrez API), bioRxiv, OpenTargets, UniProt, PDB.
- **LLM Agents**: GPT-4o / Claude 3.5 Sonnet orchestrating retrieval and scoring.
- **Knowledge Graph**: Neo4j (for mapping protein-protein interaction networks).

## Project Structure
- `src/agents/`: Multi-agent system for literature mining and target scoring.
- `src/retrieval/`: Connectors for PubMed, UniProt, and OpenTargets.
- `src/knowledge_graph/`: PPI network construction and traversal scripts.
- `app/`: Dashboard for target prioritization and evidence visualization.

## Key Features
- **Multi-Source RAG**: Cross-referencing literature evidence with experimental protein data.
- **Druggability Scoring**: Automated assessment of target binding pockets using structural data.
- **Safety Profiling**: Retrieval of known off-target effects and tissue expression patterns.
- **Mechanism-of-Action (MoA) Extraction**: LLM-guided extraction of causal links between targets and diseases.

---
Maintained by mayankbot01 | GENAI_drughunter Series — Project 3 of 10
