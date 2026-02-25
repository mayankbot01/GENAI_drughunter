# PROJECT 6 — ProteinBind-LLM: Protein-Ligand Binding Affinity Predictor

## Bottleneck Solved
Slow, expensive computational docking that still misses binding affinity

## What You Build
- Combine ESM-2 protein embeddings + ChemBERTa molecular embeddings in cross-attention binding affinity model
- Fine-tune on PDBbind + BindingDB (500K+ binding data points)
- Natural language query interface: "Find all known inhibitors of EGFR with Kd < 10nM"
- Agent autonomously queries PDB, retrieves protein structure, predicts binding for candidate molecules
- Uncertainty quantification: Monte Carlo Dropout gives confidence intervals on Kd predictions

## Tech Stack
| Component | Technology |
|---|---|
| Protein | ESM-2 (650M) embeddings via fair-esm |
| Molecule | ChemBERTa-2 / ECFP fingerprints |
| Model | Cross-attention fusion (PyTorch) |
| Dataset | PDBbind 2020 + BindingDB |
| NL Interface | LangChain + custom tool calling |
| RAG | PDB API + UniProt API |
| Deployment | FastAPI + Docker + Kubernetes |

## Project Structure
```
project_6_ProteinBind_LLM/
├── app/
│   └── dashboard.py              # NL query interface + visualization
├── configs/
├── data/
│   ├── raw/                      # PDBbind, BindingDB
│   └── processed/                # Preprocessed binding data
├── notebooks/
│   ├── 01_pdbbind_eda.ipynb
│   ├── 02_cross_attention_model.ipynb
│   └── 03_nl_query_interface.ipynb
├── src/
│   ├── models/
│   │   ├── binding_model.py      # Cross-attention fusion model
│   │   ├── esm2_encoder.py       # ESM-2 protein encoder
│   │   ├── chemberta_encoder.py  # ChemBERTa molecule encoder
│   │   └── uncertainty.py        # Monte Carlo Dropout
│   ├── agents/
│   │   └── binding_agent.py      # NL interface + PDB queries
│   ├── data/
│   │   ├── pdbbind_loader.py
│   │   └── bindingdb_loader.py
│   └── api/
│       └── main.py
├── k8s/                          # Kubernetes deployment manifests
├── tests/
├── Dockerfile
├── requirements.txt
├── MYWORK.md
└── README.md
```

## Model Architecture
```
Protein Sequence ──► ESM-2 (650M params) ──► Mean Pool ──► [1280d]
                                                               │
                                               Cross-Attention Fusion
                                                               │
Molecule SMILES ──► ChemBERTa-2 ──► [768d]                   ▼
Molecule Graph ──► ECFP4 [2048d] ──► MLP ──► [256d] ──► Regressor ──► pKd (log Kd)
                                                               │
                                               MC Dropout ──► Uncertainty (std)
```

## Natural Language Interface Examples
```
User: "Find all known inhibitors of EGFR with Kd < 10nM and selectivity over HER2"
Agent: 1. Query PDB for EGFR structures
       2. Query BindingDB for known Kd values
       3. Run binding affinity predictions on candidate molecules
       4. Filter by selectivity ratio EGFR/HER2 > 100x
       5. Return ranked list with confidence intervals

User: "Predict binding affinity of this SMILES to KRAS G12C"
Agent: 1. Fetch KRAS G12C structure (PDB: 6OIM)
       2. Compute ESM-2 embeddings for KRAS sequence
       3. Compute ChemBERTa embeddings for SMILES
       4. Predict pKd with uncertainty bounds
       5. Return: pKd = 8.3 +/- 0.4 (Kd ~5 nM [2-12 nM 95% CI])
```

## Uncertainty Quantification (Monte Carlo Dropout)
```python
def predict_with_uncertainty(model, protein_emb, mol_emb, n_samples=100):
    model.train()  # Keep dropout active
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(protein_emb, mol_emb)
            predictions.append(pred.item())
    
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    ci_95 = (mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred)
    
    return {
        'pKd_mean': mean_pred,
        'pKd_std': std_pred,
        'Kd_nM': 10 ** (-mean_pred) * 1e9,
        'confidence_interval_95': ci_95
    }
```

## Performance Targets
| Metric | Target |
|--------|--------|
| Pearson R on PDBbind test | >0.85 |
| RMSE on PDBbind | <1.2 pKd units |
| Spearman R on BindingDB | >0.80 |
| Inference time per molecule | <500ms |

## Setup & Installation
```bash
git clone https://github.com/mayankbot01/GENAI_drughunter.git
cd GENAI_drughunter/project_6_ProteinBind_LLM
pip install -r requirements.txt

# Download PDBbind data (requires registration)
python src/data/pdbbind_loader.py --version 2020

# Pre-compute ESM-2 embeddings
python src/models/esm2_encoder.py --precompute

# Train binding model
python src/models/binding_model.py --train --dataset pdbbind

# Start API
docker build -t proteinbind-llm .
docker run -p 8000:8000 proteinbind-llm
```

## Resources & References
- [ESM-2 (Meta AI)](https://github.com/facebookresearch/esm)
- [PDBbind Database](http://www.pdbbind.org.cn/)
- [BindingDB](https://www.bindingdb.org/)
- [ChemBERTa-2](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- [Monte Carlo Dropout Paper](https://arxiv.org/abs/1506.02142)
- [RCSB PDB API](https://data.rcsb.org/)
- [Kubernetes Deployment](https://kubernetes.io/docs/)

## Why It's Revolutionary
Replaces 3-day computational docking runs with sub-second neural binding affinity predictions WITH uncertainty bounds, enabling rapid screening of millions of compounds. The natural language interface makes it accessible to medicinal chemists without bioinformatics expertise.

---
*Maintained by mayankbot01 | GENAI_drughunter Series — Project 6 of 9*
