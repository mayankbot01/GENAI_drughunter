# PROJECT 5 — ClinTrialGPT: Clinical Trial Protocol Generator + Failure Risk Predictor

## Bottleneck Solved
Poorly designed trials and predictable failures wasting $500M+

## What You Build
- RAG over 400,000+ ClinicalTrials.gov records to understand what worked and what failed
- Fine-tuned LLM generates trial protocols (endpoints, dosing, inclusion/exclusion criteria)
- Failure Risk Score: trained on historical Phase 2 trial outcomes
- Comparative analysis: "How did similar molecules perform? What killed them?"
- Report generation: full PDF trial design document auto-generated

## Tech Stack
| Component | Technology |
|---|---|
| Data | ClinicalTrials.gov API + AACT database |
| RAG | LlamaIndex + pgvector (PostgreSQL) |
| Fine-tuning | Mistral-7B LoRA on trial protocol text |
| Risk Model | Gradient Boosting on trial metadata features |
| Output | LLM-generated report via structured output (Pydantic) |
| LLMOps | Prefect for pipeline orchestration + MLflow |
| Evaluation | BLEU/ROUGE on protocol quality + AUC on failure prediction |

## Project Structure
```
project_5_ClinTrialGPT/
├── app/
│   └── dashboard.py              # Protocol generator + risk dashboard
├── configs/
│   └── clintrial_config.yaml
├── data/
│   ├── raw/                      # ClinicalTrials.gov + AACT data
│   └── processed/                # Feature-engineered trial metadata
├── notebooks/
│   ├── 01_trial_data_eda.ipynb
│   ├── 02_mistral_finetuning.ipynb
│   └── 03_failure_risk_model.ipynb
├── src/
│   ├── agents/
│   │   └── protocol_agent.py     # Main LangGraph agent
│   ├── models/
│   │   ├── failure_risk_model.py # GBM failure predictor
│   │   └── mistral_finetune.py   # LoRA fine-tuning on protocol text
│   ├── rag/
│   │   ├── trial_ingestion.py    # ClinicalTrials.gov data ingestion
│   │   └── pgvector_store.py     # PostgreSQL + pgvector
│   ├── report/
│   │   └── pdf_generator.py      # Pydantic + PDF report generation
│   └── api/
│       └── main.py
├── tests/
├── Dockerfile
├── requirements.txt
├── MYWORK.md
└── README.md
```

## Trial Failure Prediction Features
```python
# Feature engineering from ClinicalTrials.gov metadata
features = [
    'phase',                    # 1/2/3
    'enrollment_size',          # planned vs actual
    'duration_months',          # trial length
    'num_primary_endpoints',    # endpoint count
    'endpoint_type',            # surrogate vs clinical
    'mechanism_of_action',      # encoded MOA class
    'target_class',             # kinase/GPCR/NHR etc.
    'indication',               # oncology/CNS/metabolic
    'prior_phase1_success',     # Phase 1 safety outcomes
    'competitor_failures',      # same target failures
    'biomarker_selection',      # biomarker-selected trial?
    'adaptive_design',          # adaptive/traditional
    'sponsor_type',             # pharma/biotech/academia
    'primary_endpoint_timing',  # months to primary endpoint
    'num_sites'                 # global vs regional
]
```

## Protocol Generation Pipeline
```
[Disease + Molecule Input]
    ↓
[ClinTrials RAG] → Retrieve similar trials
    ↓
[Failure Analysis] → "What killed similar trials?"
    ↓
[Mistral-7B LoRA] → Generate optimized protocol draft
    ↓
[Risk Scoring] → Gradient Boosting failure probability
    ↓
[Pydantic Validation] → Structured protocol object
    ↓
[PDF Report Generator] → Full trial design document
```

## Pydantic Protocol Schema
```python
class TrialProtocol(BaseModel):
    trial_name: str
    phase: Literal['Phase 1', 'Phase 2', 'Phase 3']
    indication: str
    primary_endpoint: str
    secondary_endpoints: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    dosing_regimen: DosingSche dule
    enrollment_target: int
    duration_weeks: int
    biomarker_strategy: Optional[str]
    adaptive_design: bool
    safety_stopping_rules: List[str]
    failure_risk_score: float  # 0-1
    failure_risk_factors: List[str]
    historical_comparators: List[str]
```

## Data Sources
- **ClinicalTrials.gov**: 400,000+ registered trials via API/AACT database
- **FDA Clinical Pharmacology DB**: Approved drug protocols
- **PubMed**: Clinical trial publications
- **AACT (Aggregate Analysis of ClinicalTrials.gov)**: PostgreSQL dump

## Key Metrics & Performance Targets
| Metric | Target |
|--------|--------|
| Failure prediction AUC | >0.75 |
| Protocol BLEU vs human-written | >0.60 |
| Protocol ROUGE-L | >0.65 |
| RAG retrieval time | <3 seconds |
| PDF generation time | <30 seconds |

## Setup & Installation
```bash
git clone https://github.com/mayankbot01/GENAI_drughunter.git
cd GENAI_drughunter/project_5_ClinTrialGPT
pip install -r requirements.txt

# Set up PostgreSQL + pgvector
docker-compose up -d postgres

# Ingest ClinicalTrials.gov data
python src/rag/trial_ingestion.py --max-trials 50000

# Train failure risk model
python src/models/failure_risk_model.py --train

# Fine-tune Mistral-7B (requires GPU)
python src/models/mistral_finetune.py --base mistralai/Mistral-7B-v0.1

# Launch API
uvicorn src.api.main:app --reload
```

## Resources & References
- [ClinicalTrials.gov API](https://clinicaltrials.gov/api/)
- [AACT Database](https://aact.ctti-clinicaltrials.org/)
- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [pgvector](https://github.com/pgvector/pgvector)
- [LlamaIndex](https://docs.llamaindex.ai/)
- [Prefect](https://docs.prefect.io/)
- [MLflow](https://mlflow.org/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## Why It's Revolutionary
The first AI system that not only writes clinical trial protocols but PREDICTS their probability of failure before a single patient is enrolled. By learning from 400K+ historical trials, it identifies the exact design elements that killed previous programs.

---
*Maintained by mayankbot01 | GENAI_drughunter Series — Project 5 of 9*
