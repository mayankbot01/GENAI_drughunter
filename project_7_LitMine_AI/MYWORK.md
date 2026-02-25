# PROJECT 7 — LitMine-AI: Autonomous Drug Literature Intelligence System

## Bottleneck Solved
Researchers missing critical papers — 5,000+ biomedical papers published daily

## What You Build
- Automated ingestion pipeline: pulls from PubMed, bioRxiv, ChEMBL daily via APIs
- Hierarchical RAG: coarse retrieval (BM25) → fine reranking (ColBERT/BGE-reranker) → LLM synthesis
- Contradiction detector: finds conflicting claims across papers about the same drug/target
- Novelty scorer: given your molecule, finds how novel it is vs existing patent and literature space
- Alert system: daily digest email — "3 new papers on your target PCSK9 published today"

## Tech Stack
| Component | Technology |
|---|---|
| Ingestion | PubMed API + arXiv API + Airflow DAG |
| Embeddings | BioBERT / PubMedBERT + FAISS |
| Reranker | BGE-reranker-large |
| RAG | LlamaIndex advanced RAG with HyDE |
| LLM | Mistral/Llama-3 for synthesis |
| Contradiction Detection | NLI model (DeBERTa fine-tuned on BioNLI) |
| Monitoring | LangFuse + custom eval suite |

## Project Structure
```
project_7_LitMine_AI/
├── app/
│   └── dashboard.py              # Literature search + alert dashboard
├── configs/
│   └── litmine_config.yaml
├── data/
│   ├── raw/                      # Ingested papers
│   └── index/                    # FAISS vector index
├── notebooks/
│   ├── 01_pubmed_ingestion.ipynb
│   ├── 02_hierarchical_rag.ipynb
│   └── 03_contradiction_detection.ipynb
├── src/
│   ├── ingestion/
│   │   ├── pubmed_fetcher.py     # PubMed E-utils API
│   │   ├── arxiv_fetcher.py      # arXiv API
│   │   └── chembl_fetcher.py     # ChEMBL REST API
│   ├── rag/
│   │   ├── hierarchical_rag.py   # BM25 + reranker pipeline
│   │   ├── hyde_rag.py           # HyDE (Hypothetical Doc Embeddings)
│   │   └── faiss_store.py        # FAISS vector store management
│   ├── analysis/
│   │   ├── contradiction_detector.py  # DeBERTa NLI model
│   │   ├── novelty_scorer.py     # Novelty vs literature space
│   │   └── synthesis_agent.py    # LLM synthesis of retrieved papers
│   ├── alerts/
│   │   └── daily_digest.py       # Email alert system
│   └── pipeline/
│       └── airflow_dag.py        # Airflow DAG for daily ingestion
├── tests/
├── Dockerfile
├── requirements.txt
├── MYWORK.md
└── README.md
```

## Hierarchical RAG Architecture
```
[Query: "PCSK9 inhibitor clinical trials 2024"]
    ↓
[HyDE] → Generate hypothetical abstract → Embed
    ↓
[BM25 Coarse Retrieval] → Top-100 candidates (keyword match)
    ↓
[BioBERT Dense Retrieval] → Top-50 (semantic similarity)
    ↓
[BGE Reranker] → Top-10 (cross-encoder reranking)
    ↓
[LLM Synthesis] → Coherent answer with citations
    ┣━━► [Contradiction Detector] → Flag conflicting claims
    └━━► [Novelty Scorer] → How novel is this claim?
```

## Contradiction Detection
```python
# DeBERTa NLI model for biomedical claim contradiction
class BioContradictionDetector:
    def __init__(self):
        # Fine-tuned on BioNLI, MedNLI datasets
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'cross-encoder/nli-deberta-v3-large'
        )
    
    def detect(self, claim1: str, claim2: str) -> dict:
        """
        Returns: {label: ENTAILMENT/CONTRADICTION/NEUTRAL, confidence: float}
        """
        inputs = self.tokenizer(claim1, claim2, return_tensors='pt')
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        labels = ['CONTRADICTION', 'NEUTRAL', 'ENTAILMENT']
        return {
            'label': labels[probs.argmax()],
            'confidence': probs.max().item()
        }
```

## Novelty Scoring
```python
def score_novelty(molecule_smiles: str, query: str, top_k: int = 50) -> dict:
    """
    Scores molecular novelty against:
    1. PubMed literature (publication distance)
    2. Patent space (USPTO + EPO)
    3. ChEMBL database (bioactivity data)
    """
    # Semantic distance from nearest papers
    lit_distance = faiss_store.max_inner_product_search(query_embedding)
    # Structural novelty via Tanimoto similarity
    structural_novelty = 1 - max_tanimoto_similarity(molecule_smiles, reference_mols)
    
    return {
        'literature_novelty': lit_distance,
        'structural_novelty': structural_novelty,
        'combined_novelty': 0.5 * lit_distance + 0.5 * structural_novelty,
        'nearest_papers': nearest_papers
    }
```

## Daily Digest System
- Airflow DAG runs at 6 AM daily
- Fetches all new papers since last run (PubMed + arXiv)
- Routes papers to relevant research topics via embedding similarity
- Generates personalized digest per researcher per topic
- Sends HTML email with: title, abstract summary, key findings, contradiction alerts

## Key APIs & Data Sources
| Source | Access | Volume |
|--------|--------|--------|
| PubMed | E-utilities API | 36M+ papers |
| bioRxiv | API | 200K+ preprints |
| ChEMBL | REST API | 2.2M compounds |
| USPTO Patents | Full-text Search | 12M+ patents |
| ClinicalTrials | API | 400K+ trials |

## Performance Targets
| Metric | Target |
|--------|--------|
| Retrieval latency (P95) | <2 seconds |
| BM25 recall@10 | >0.80 |
| Reranked NDCG@10 | >0.85 |
| Contradiction detection F1 | >0.82 |
| Daily ingestion throughput | 5,000+ papers/day |

## Setup & Installation
```bash
git clone https://github.com/mayankbot01/GENAI_drughunter.git
cd GENAI_drughunter/project_7_LitMine_AI
pip install -r requirements.txt

# Start Airflow
docker-compose up airflow-init
docker-compose up

# Initial data ingestion
python src/ingestion/pubmed_fetcher.py --years 5 --query "drug discovery"

# Build FAISS index
python src/rag/faiss_store.py --build

# Run search
python src/rag/hierarchical_rag.py --query "PCSK9 inhibitors Phase 2"
```

## Resources & References
- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- [BGE-Reranker](https://huggingface.co/BAAI/bge-reranker-large)
- [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1)
- [LlamaIndex HyDE](https://docs.llamaindex.ai/en/stable/examples/query_transformations/HyDEQueryTransformDemo/)
- [Apache Airflow](https://airflow.apache.org/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangFuse](https://langfuse.com/)
- [DeBERTa NLI](https://huggingface.co/cross-encoder/nli-deberta-v3-large)

## Why It's Revolutionary
First system that not only retrieves papers but DETECTS CONTRADICTIONS between them and scores molecular NOVELTY vs the entire literature space. Gives researchers a daily curated briefing that takes 6 hours to do manually in under 2 minutes.

---
*Maintained by mayankbot01 | GENAI_drughunter Series — Project 7 of 9*
