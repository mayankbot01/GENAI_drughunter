# PROJECT 8 — SideEffect-Navigator: Post-Market Drug Safety Signal Detection

## Bottleneck Solved
Safety signals missed until Phase 3 or post-market, killing programs late

## What You Build
- Disproportionality analysis on FAERS (FDA Adverse Event Reporting System) — 15M+ reports
- LLM layer: explains biological mechanism behind unexpected safety signals
- Real-time Twitter/Reddit/patient forum scraper: detects patient-reported side effects early
- Structural alert cross-reference: links detected safety signals back to molecular substructures
- Dashboard: interactive signal monitoring with drill-down to individual case reports

## Tech Stack
| Component | Technology |
|---|---|
| Data | FAERS + SIDER database + social media APIs |
| Analysis | PRR, ROR disproportionality (statistical signal detection) |
| NLP | BioBERT NER for adverse event extraction |
| LLM | GPT-4o / Claude for mechanistic explanation |
| RAG | DrugBank + OMIM for mechanism grounding |
| Frontend | Plotly Dash + real-time WebSocket updates |
| LLMOps | Evidently AI for data drift monitoring |

## Project Structure
```
project_8_SideEffect_Navigator/
├── app/
│   ├── dashboard.py              # Plotly Dash real-time dashboard
│   └── websocket_server.py       # Real-time signal updates
├── configs/
│   └── safety_config.yaml
├── data/
│   ├── faers/                    # FAERS quarterly downloads
│   ├── sider/                    # SIDER side effect database
│   └── social/                   # Patient forum data
├── notebooks/
│   ├── 01_faers_eda.ipynb
│   ├── 02_disproportionality.ipynb
│   └── 03_biobert_ner.ipynb
├── src/
│   ├── analysis/
│   │   ├── disproportionality.py # PRR + ROR calculations
│   │   └── signal_detector.py    # Automated signal detection
│   ├── nlp/
│   │   ├── biobert_ner.py        # Named entity recognition
│   │   └── social_scraper.py     # Patient forum NLP
│   ├── agents/
│   │   └── mechanism_agent.py    # LLM mechanistic explanation
│   ├── rag/
│   │   └── drugbank_rag.py       # DrugBank + OMIM RAG
│   └── api/
│       └── main.py
├── tests/
├── Dockerfile
├── requirements.txt
├── MYWORK.md
└── README.md
```

## Disproportionality Analysis Methods
```python
class DisproportionalityAnalyzer:
    """
    Implements standard pharmacovigilance signal detection methods:
    - PRR: Proportional Reporting Ratio
    - ROR: Reporting Odds Ratio  
    - IC: Information Component (Bayesian)
    - EBGM: Empirical Bayes Geometric Mean (FDA method)
    """

    def compute_prr(self, drug: str, event: str, data: pd.DataFrame) -> dict:
        """
        PRR = (a/b) / (c/d)
        a: reports of drug + event
        b: reports of drug + NOT event
        c: reports of NOT drug + event
        d: reports of NOT drug + NOT event
        Signal threshold: PRR > 2 AND chi2 > 4 AND n >= 3
        """
        a = len(data[(data.drug == drug) & (data.event == event)])
        b = len(data[(data.drug == drug) & (data.event != event)])
        c = len(data[(data.drug != drug) & (data.event == event)])
        d = len(data[(data.drug != drug) & (data.event != event)])

        prr = (a / (a + b)) / (c / (c + d)) if (a + b) > 0 and (c + d) > 0 else 0
        ror = (a * d) / (b * c) if b > 0 and c > 0 else 0

        # Signal criteria (WHO Uppsala Monitoring Centre)
        is_signal = prr >= 2 and a >= 3

        return {
            'drug': drug, 'event': event,
            'n_reports': a, 'prr': round(prr, 3),
            'ror': round(ror, 3), 'is_signal': is_signal
        }
```

## LLM Mechanism Explanation Pipeline
```
[Detected Safety Signal: Drug X + Cardiac Event]
    ↓
[RAG: DrugBank + OMIM]
    │  → Drug X mechanism of action
    │  → hERG channel interactions
    │  → Known cardiac toxicity pathways
    ▼
[GPT-4o / Claude]
    → "Drug X inhibits hERG potassium channels...
       ...prolonging QT interval...leading to torsades de pointes...
       ...mechanism confirmed by crystal structure analysis..."
    ↓
[Structural Alert Cross-Reference]
    → "Flagged: aromatic amine substructure at C5 position"
```

## Social Media AE Detection
```python
# Extract adverse events from patient posts using BioBERT NER
ner_pipeline = pipeline(
    'ner',
    model='allenai/scibert_scivocab_uncased',
    aggregation_strategy='simple'
)

# Patient post: "After taking Drug X for 3 days, developed severe headache and blurry vision"
# Extracted: [('Drug X', 'DRUG'), ('headache', 'ADE'), ('blurry vision', 'ADE')]
# MedDRA mapping: headache -> PT: 10019211, blurry vision -> PT: 10047513
```

## Key Data Sources
| Source | Content | Size |
|--------|---------|------|
| FDA FAERS | Adverse event reports | 15M+ reports |
| SIDER | Known drug side effects | 1,430 drugs |
| DrugBank | Drug mechanisms | 14,000+ drugs |
| OMIM | Gene-disease associations | 7,000+ genes |
| MedDRA | Adverse event taxonomy | 25,000+ terms |
| EudraVigilance | EU adverse events | 10M+ reports |

## Performance Targets
| Metric | Target |
|--------|--------|
| Signal sensitivity (vs known signals) | >0.85 |
| False positive rate | <0.15 |
| NER F1 on adverse events | >0.82 |
| Dashboard refresh latency | <5 seconds |

## Setup & Installation
```bash
git clone https://github.com/mayankbot01/GENAI_drughunter.git
cd GENAI_drughunter/project_8_SideEffect_Navigator
pip install -r requirements.txt

# Download FAERS data (quarterly)
python src/analysis/signal_detector.py --download-faers --years 2020-2024

# Run disproportionality analysis
python src/analysis/disproportionality.py --drug aspirin --topk 20

# Start real-time dashboard
python app/dashboard.py
```

## Resources & References
- [FDA FAERS Database](https://www.fda.gov/drugs/surveillance/fda-adverse-event-reporting-system-faers)
- [SIDER Database](http://sideeffects.embl.de/)
- [WHO Signal Detection Methods](https://www.who.int/docs/default-source/pharmacovigilance/signal-detection.pdf)
- [BioBERT NER](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
- [MedDRA Dictionary](https://www.meddra.org/)
- [Plotly Dash](https://dash.plotly.com/)
- [Evidently AI](https://docs.evidentlyai.com/)
- [EudraVigilance API](https://eudravigilance.ema.europa.eu/)

## Why It's Revolutionary
Combines FDA's 15M+ adverse event reports with real-time patient social media data AND mechanistic AI reasoning to detect safety signals weeks before they appear in clinical data. The structural alert cross-referencing links every signal back to the exact molecular moiety responsible.

---
*Maintained by mayankbot01 | GENAI_drughunter Series — Project 8 of 9*
