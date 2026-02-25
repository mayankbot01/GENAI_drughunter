from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal

app = FastAPI(title="ClinTrialGPT API")

class DosingSchedule(BaseModel):
    dose_amount: str
    frequency: str
    route: str

class TrialProtocol(BaseModel):
    trial_name: str
    phase: Literal['Phase 1', 'Phase 2', 'Phase 3']
    indication: str
    primary_endpoint: str
    secondary_endpoints: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    dosing_regimen: DosingSchedule
    enrollment_target: int
    duration_weeks: int
    biomarker_strategy: Optional[str]
    adaptive_design: bool
    safety_stopping_rules: List[str]
    failure_risk_score: float # 0-1
    failure_risk_factors: List[str]
    historical_comparators: List[str]

@app.get("/")
async def root():
    return {"message": "Welcome to ClinTrialGPT Protocol Generator"}

@app.post("/generate-protocol", response_model=TrialProtocol)
async def generate_protocol(indication: str, molecule_description: str):
    # Placeholder for LLM generation logic
    return {
        "trial_name": f"Study of {molecule_description} in {indication}",
        "phase": "Phase 2",
        "indication": indication,
        "primary_endpoint": "Change from baseline in clinical score at 24 weeks",
        "secondary_endpoints": ["Safety", "PK/PD"],
        "inclusion_criteria": ["Age > 18", "Diagnosed with condition"],
        "exclusion_criteria": ["Pregnant", "Prior treatment"],
        "dosing_regimen": {"dose_amount": "100mg", "frequency": "Daily", "route": "Oral"},
        "enrollment_target": 250,
        "duration_weeks": 52,
        "biomarker_strategy": "High expression of Target X",
        "adaptive_design": True,
        "safety_stopping_rules": ["Serious adverse events > 10%"],
        "failure_risk_score": 0.45,
        "failure_risk_factors": ["High dropout rate in similar trials"],
        "historical_comparators": ["Trial-ABC-123"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
