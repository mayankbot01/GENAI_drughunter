# PROJECT 4 — ADMET-Oracle: Agentic ADMET Profiling System

## Bottleneck Solved
Poor pharmacokinetics killing Phase 1/2 candidates silently

## What You Build
- Multi-agent system: 5 specialized sub-agents (Absorption, Distribution, Metabolism, Excretion, Toxicity), each fine-tuned or RAG-equipped
- Orchestrator agent routes molecule queries to correct sub-agents
- Each agent returns: predicted value + confidence + supporting literature + structural alerts
- Batch pipeline: accepts SDF/CSV of compound libraries, profiles all at once, ranks by ADMET score
- MCP integration: expose as a tool that any drug design LLM agent can call

## Tech Stack
| Component | Technology |
|---|---|
| Agent Framework | LangGraph (stateful multi-agent) |
| Property Models | chemprop (D-MPNN) for each ADMET endpoint |
| Fine-tuning | LoRA on ADMET-specific datasets |
| Orchestration | LangGraph supervisor pattern |
| Serving | Ray Serve for parallel agent execution |
| Monitoring | Arize Phoenix + custom dashboards |

## ADMET Sub-Agents

### Absorption Agent
- Predicts: Caco-2 permeability, HIA (human intestinal absorption), oral bioavailability (%F)
- Datasets: PermeabilityDB, ZINC-ADMET
- Alerts: P-gp substrate flags, efflux transporter recognition

### Distribution Agent
- Predicts: BBB penetration, plasma protein binding (PPB), volume of distribution (Vd)
- Datasets: BBB dataset (B3DB), PPB dataset
- Alerts: CNS MPO score, P-gp efflux risk

### Metabolism Agent
- Predicts: CYP450 inhibition (1A2, 2C9, 2C19, 2D6, 3A4), half-life (t1/2), microsomal stability
- Datasets: ChEMBL CYP data, CYP450 2D6 dataset
- Alerts: Michael acceptors, reactive metabolite precursors

### Excretion Agent
- Predicts: Renal clearance, biliary excretion, t1/2
- Datasets: OATP substrate datasets, renal clearance DB
- Alerts: Transporter substrate/inhibitor flags

### Toxicity Agent
- Predicts: hERG inhibition (cardiac), DILI, AMES mutagenicity
- Datasets: hERG central, DILIst, Ames dataset
- Alerts: SMARTS-based structural alerts

## Project Structure
```
project_4_ADMET_Oracle/
├── app/
│   └── dashboard.py              # Batch ADMET dashboard
├── configs/
│   └── admet_config.yaml
├── data/
│   ├── raw/                      # All ADMET datasets
│   └── processed/
├── notebooks/
│   ├── 01_admet_eda.ipynb
│   ├── 02_chemprop_training.ipynb
│   └── 03_lora_finetuning.ipynb
├── src/
│   ├── agents/
│   │   ├── orchestrator.py       # Supervisor LangGraph agent
│   │   ├── absorption_agent.py
│   │   ├── distribution_agent.py
│   │   ├── metabolism_agent.py
│   │   ├── excretion_agent.py
│   │   └── toxicity_agent.py
│   ├── models/
│   │   ├── chemprop_models.py    # D-MPNN for each endpoint
│   │   └── lora_finetune.py      # LoRA fine-tuning pipeline
│   ├── batch/
│   │   └── batch_processor.py    # SDF/CSV batch processing
│   ├── mcp/
│   │   └── mcp_server.py         # MCP tool exposure
│   └── api/
│       └── main.py               # FastAPI + Ray Serve
├── tests/
├── Dockerfile
├── requirements.txt
├── MYWORK.md
└── README.md
```

## Multi-Agent Architecture
```
[Molecule SMILES / SDF Library]
        ↓
[Orchestrator Agent (LangGraph Supervisor)]
    ├───► [Absorption Agent] → F%, Caco-2, HIA
    ├───► [Distribution Agent] → BBB, PPB, Vd
    ├───► [Metabolism Agent] → CYP450, t1/2
    ├───► [Excretion Agent] → Clearance, CLren
    └───► [Toxicity Agent] → hERG, DILI, AMES
        ↓
[Aggregator] → Overall ADMET Score + Lipinski/Veber flags
        ↓
[LLM Report Generator] → Plain-English ADMET report
        ↓
[MCP Tool] → Expose to external drug design agents
```

## ADMET Scoring System
```python
admet_score = (
    0.20 * normalize(absorption_score) +
    0.20 * normalize(distribution_score) +
    0.20 * normalize(metabolism_score) +
    0.15 * normalize(excretion_score) +
    0.25 * normalize(toxicity_score)  # Toxicity weighted highest
)
# Score: 0-1, where 1 = ideal ADMET profile
```

## Key Datasets
| Endpoint | Dataset | Size |
|----------|---------|------|
| Caco-2 | Wang et al. | 906 compounds |
| BBB | B3DB | 7,807 compounds |
| hERG | hERG Central | 306,893 compounds |
| CYP1A2 | ChEMBL | 12,000+ compounds |
| DILI | DILIst | 1,145 compounds |
| AMES | Ames test dataset | 6,512 compounds |
| Lipophilicity | ChEMBL | 4,200 compounds |
| Solubility (ESOL) | Delaney | 1,128 compounds |

## Setup & Installation
```bash
git clone https://github.com/mayankbot01/GENAI_drughunter.git
cd GENAI_drughunter/project_4_ADMET_Oracle
pip install -r requirements.txt

# Train all ADMET models
python src/models/chemprop_models.py --train-all

# Start Ray Serve
ray start --head
python src/api/main.py

# Batch process a library
python src/batch/batch_processor.py --input data/library.sdf --output results.csv

# Launch dashboard
streamlit run app/dashboard.py
```

## Resources & References
- [chemprop Documentation](https://chemprop.readthedocs.io/)
- [DeepChem ADMET](https://deepchem.io/tutorials/the-tox21-dataset/)
- [ESOL Dataset](https://pubs.acs.org/doi/10.1021/ci034243x)
- [B3DB (BBB)](https://github.com/theochem/B3DB)
- [hERG Central](http://www.hergcentral.org/)
- [LangGraph Supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [Arize Phoenix](https://phoenix.arize.com/)

## Why It's Revolutionary
First agentic ADMET system where each endpoint has a specialized fine-tuned agent with RAG-grounded literature evidence, all orchestrated by a supervisor LLM that can explain trade-offs and suggest compound series pivots. Exposed as an MCP tool for seamless integration into multi-step drug discovery pipelines.

---
*Maintained by mayankbot01 | GENAI_drughunter Series — Project 4 of 9*
