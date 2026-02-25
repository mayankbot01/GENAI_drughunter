# PROJECT 9 — RepurposeAI: Drug Repurposing Engine using Multi-Hop Graph Reasoning

## Bottleneck Solved
Ignoring $1B+ already-approved drugs that could treat new diseases

## What You Build
- Biomedical Knowledge Graph: Drug → Protein → Pathway → Disease (from DrugBank, OMIM, Reactome, DisGeNET)
- Graph Neural Network: link prediction to find novel Drug-Disease edges
- LLM Multi-Hop Reasoning: "Why might Metformin treat Alzheimer's?" → AMPK → mTOR → neuroinflammation → Aβ clearance
- Validation layer: cross-references predictions against clinical trials and genetic evidence
- Output: ranked repurposing candidates with mechanistic rationale + clinical feasibility score

## Tech Stack
| Component | Technology |
|---|---|
| KG Construction | Neo4j + RDFLib |
| GNN | PyTorch Geometric (R-GCN for heterogeneous graphs) |
| LLM Reasoning | LangChain + LangGraph multi-hop agent |
| KG Embeddings | TransE / RotatE |
| Validation | OpenTargets API + ClinicalTrials.gov |
| Evaluation | Hit@K, MRR on held-out drug-disease pairs |
| Serving | GraphQL API + Streamlit frontend |

## Project Structure
```
project_9_RepurposeAI/
├── app/
│   └── dashboard.py              # Streamlit repurposing dashboard
├── configs/
│   └── repurpose_config.yaml
├── data/
│   ├── raw/                      # DrugBank, OMIM, Reactome, DisGeNET
│   └── kg/                       # Processed KG triples
├── notebooks/
│   ├── 01_kg_construction.ipynb
│   ├── 02_rcgn_training.ipynb
│   └── 03_multihop_reasoning.ipynb
├── src/
│   ├── kg/
│   │   ├── kg_builder.py         # KG construction from multiple sources
│   │   └── neo4j_loader.py       # Load KG into Neo4j
│   ├── models/
│   │   ├── rcgn.py               # Relational GCN (PyG)
│   │   ├── transe.py             # TransE KG embeddings
│   │   └── link_predictor.py     # Drug-Disease link prediction
│   ├── agents/
│   │   └── multihop_agent.py     # LangGraph multi-hop reasoning
│   ├── validation/
│   │   └── clinical_validator.py # OpenTargets + ClinTrials validation
│   └── api/
│       └── graphql_api.py        # GraphQL API
├── tests/
├── Dockerfile
├── requirements.txt
├── MYWORK.md
└── README.md
```

## Knowledge Graph Schema
```
Node Types:
  Drug (DrugBank ID, SMILES, MOA)
  Protein (UniProt ID, gene name, function)
  Disease (OMIM ID, ICD10, MeSH term)
  Pathway (Reactome ID, name)
  Gene (Entrez ID, expression data)

Edge Types (Heterogeneous):
  Drug --[TARGETS]--> Protein          (DrugBank)
  Drug --[TREATS]--> Disease           (DrugBank, known indications)
  Protein --[INTERACTS_WITH]--> Protein (STRING PPI)
  Protein --[IN_PATHWAY]--> Pathway     (Reactome)
  Gene --[ASSOCIATED_WITH]--> Disease  (DisGeNET)
  Pathway --[LINKED_TO]--> Disease     (Reactome-Disease)
```

## R-GCN Link Prediction
```python
class DrugRepurposeRCGN(nn.Module):
    """
    Relational Graph Convolutional Network for Drug-Disease link prediction.
    Handles heterogeneous KG with multiple edge types.
    """
    def __init__(self, num_nodes, num_relations, hidden_dim=256, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations)
            for _ in range(num_layers)
        ])
        # Link prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_type):
        h = self.embedding(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_type))
        return h
    
    def predict_link(self, drug_id, disease_id, node_embeddings):
        """Predict probability of Drug-Disease treatment relationship."""
        drug_emb = node_embeddings[drug_id]
        disease_emb = node_embeddings[disease_id]
        combined = torch.cat([drug_emb, disease_emb], dim=-1)
        return self.predictor(combined)
```

## Multi-Hop LLM Reasoning Example
```
Query: "Why might Metformin treat Alzheimer's disease?"

Agent traces the knowledge graph:
Step 1: Metformin --[TARGETS]--> AMPK (AMP-activated protein kinase)
Step 2: AMPK --[INHIBITS]--> mTOR (mechanistic target of rapamycin)
Step 3: mTOR --[REGULATES]--> Autophagy pathway
Step 4: Autophagy --[CLEARS]--> Amyloid-beta (Aβ plaques)
Step 5: Aβ --[CAUSES]--> Neuroinflammation
Step 6: Neuroinflammation --[DRIVES]--> Alzheimer's disease

Conclusion: Metformin activates AMPK → suppresses mTOR → 
enhances autophagy → clears Aβ plaques → reduces neuroinflammation

Validation: 3 ClinicalTrials.gov studies (NCT03451084, NCT02432287...)
           OpenTargets genetic evidence score: 0.67
           Clinical feasibility: HIGH (approved drug, known safety profile)
```

## Repurposing Scoring System
```python
repurpose_score = (
    0.30 * gnn_link_probability +      # R-GCN predicted score
    0.25 * path_confidence_score +     # Multi-hop path quality
    0.20 * genetic_evidence_score +    # OpenTargets genetic validation
    0.15 * clinical_trial_evidence +   # Existing trial evidence
    0.10 * safety_profile_score        # Known safety from original indication
)
```

## Key Data Sources
| Source | Content | Access |
|--------|---------|--------|
| DrugBank | Drug-target-disease | XML download |
| OMIM | Disease-gene | API |
| Reactome | Biological pathways | API |
| DisGeNET | Gene-disease associations | REST API |
| STRING | PPI network | Download |
| OpenTargets | Genetic evidence | GraphQL API |
| ClinicalTrials.gov | Trial validation | API |

## Evaluation Metrics
| Metric | Target |
|--------|--------|
| Hits@1 | >0.25 |
| Hits@10 | >0.65 |
| MRR | >0.35 |
| Known repurposing recall@50 | >0.80 |

## Setup & Installation
```bash
git clone https://github.com/mayankbot01/GENAI_drughunter.git
cd GENAI_drughunter/project_9_RepurposeAI
pip install -r requirements.txt

# Build Knowledge Graph
python src/kg/kg_builder.py --sources drugbank,omim,reactome,disgenet
python src/kg/neo4j_loader.py

# Train R-GCN model
python src/models/rcgn.py --train --epochs 200

# Run repurposing query
python src/agents/multihop_agent.py --drug Metformin --disease Alzheimer

# Launch dashboard
streamlit run app/dashboard.py
```

## Resources & References
- [DrugBank Database](https://go.drugbank.com/)
- [Reactome Pathway DB](https://reactome.org/)
- [DisGeNET](https://www.disgenet.org/)
- [OpenTargets Platform](https://platform.opentargets.org/)
- [R-GCN Paper](https://arxiv.org/abs/1703.06103)
- [TransE Paper](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Strawberry Fields: Drug Repurposing Review](https://www.nature.com/articles/s41573-019-0024-5)

## Why It's Revolutionary
Combines GNN link prediction with explainable multi-hop LLM reasoning to not just predict BUT EXPLAIN repurposing candidates with full mechanistic biological rationale. Unlike black-box similarity methods, every prediction comes with a step-by-step biological story validated against genetic evidence and clinical trials.

---
*Maintained by mayankbot01 | GENAI_drughunter Series — Project 9 of 9*
