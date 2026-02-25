"""
ADMET-Oracle: Orchestrator Agent
Supervisor LangGraph agent that coordinates all 5 ADMET sub-agents.
"""

from typing import TypedDict, List, Optional, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor


class ADMETState(TypedDict):
    smiles: str
    molecule_name: str
    absorption: Optional[Dict]
    distribution: Optional[Dict]
    metabolism: Optional[Dict]
    excretion: Optional[Dict]
    toxicity: Optional[Dict]
    admet_score: Optional[float]
    lipinski_pass: Optional[bool]
    veber_pass: Optional[bool]
    structural_alerts: Optional[List[str]]
    llm_report: Optional[str]
    recommendation: Optional[str]  # ADVANCE / OPTIMIZE / REJECT


class ADMETOrchestrator:
    """
    Supervisor LangGraph agent for ADMET profiling.
    Runs 5 specialized sub-agents in parallel, then synthesizes.
    """

    def __init__(
        self,
        absorption_agent,
        distribution_agent,
        metabolism_agent,
        excretion_agent,
        toxicity_agent,
        llm_model: str = "gpt-4o",
        parallel: bool = True
    ):
        self.agents = {
            "absorption": absorption_agent,
            "distribution": distribution_agent,
            "metabolism": metabolism_agent,
            "excretion": excretion_agent,
            "toxicity": toxicity_agent
        }
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.parallel = parallel
        self.workflow = self._build_graph()

    def _build_graph(self):
        wf = StateGraph(ADMETState)

        wf.add_node("run_admet", self.run_all_agents)
        wf.add_node("lipinski_check", self.check_drug_likeness)
        wf.add_node("score_admet", self.compute_admet_score)
        wf.add_node("generate_report", self.generate_llm_report)
        wf.add_node("recommend", self.make_recommendation)

        wf.set_entry_point("run_admet")
        wf.add_edge("run_admet", "lipinski_check")
        wf.add_edge("lipinski_check", "score_admet")
        wf.add_edge("score_admet", "generate_report")
        wf.add_edge("generate_report", "recommend")
        wf.add_edge("recommend", END)

        return wf.compile()

    def run_all_agents(self, state: ADMETState) -> ADMETState:
        """Run all 5 ADMET agents (parallel or sequential)."""
        smiles = state["smiles"]

        if self.parallel:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    name: executor.submit(agent.predict, smiles)
                    for name, agent in self.agents.items()
                }
                results = {name: future.result() for name, future in futures.items()}
        else:
            results = {name: agent.predict(smiles) for name, agent in self.agents.items()}

        # Collect all structural alerts
        all_alerts = []
        for result in results.values():
            all_alerts.extend(result.get("alerts", []))

        return {
            **state,
            "absorption": results["absorption"],
            "distribution": results["distribution"],
            "metabolism": results["metabolism"],
            "excretion": results["excretion"],
            "toxicity": results["toxicity"],
            "structural_alerts": list(set(all_alerts))
        }

    def check_drug_likeness(self, state: ADMETState) -> ADMETState:
        """Check Lipinski Rule of 5 and Veber rules."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors

            mol = Chem.MolFromSmiles(state["smiles"])
            if mol:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = rdMolDescriptors.CalcNumHBD(mol)
                hba = rdMolDescriptors.CalcNumHBA(mol)
                rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
                tpsa = Descriptors.TPSA(mol)

                lipinski = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
                veber = (rotbonds <= 10) and (tpsa <= 140)

                return {**state, "lipinski_pass": lipinski, "veber_pass": veber}
        except Exception:
            pass
        return {**state, "lipinski_pass": None, "veber_pass": None}

    def compute_admet_score(self, state: ADMETState) -> ADMETState:
        """Compute weighted ADMET score."""
        weights = {"absorption": 0.20, "distribution": 0.20,
                   "metabolism": 0.20, "excretion": 0.15, "toxicity": 0.25}

        total_score = 0.0
        for component, weight in weights.items():
            comp_data = state.get(component, {})
            if comp_data:
                comp_score = comp_data.get("score", 0.5)
                total_score += weight * comp_score

        # Penalty for structural alerts
        alert_penalty = min(len(state.get("structural_alerts", [])) * 0.05, 0.3)
        final_score = max(0, total_score - alert_penalty)

        return {**state, "admet_score": round(final_score, 3)}

    def generate_llm_report(self, state: ADMETState) -> ADMETState:
        """LLM synthesizes ADMET report."""
        prompt = f"""
You are an expert pharmacokineticist. Generate a comprehensive ADMET report.

Molecule: {state['molecule_name']} ({state['smiles']})

ADMET Predictions:
- Absorption: {json.dumps(state['absorption'], indent=2)}
- Distribution: {json.dumps(state['distribution'], indent=2)}
- Metabolism: {json.dumps(state['metabolism'], indent=2)}
- Excretion: {json.dumps(state['excretion'], indent=2)}
- Toxicity: {json.dumps(state['toxicity'], indent=2)}

Drug-likeness:
- Lipinski RO5: {'PASS' if state.get('lipinski_pass') else 'FAIL'}
- Veber Rules: {'PASS' if state.get('veber_pass') else 'FAIL'}

Structural Alerts: {state.get('structural_alerts', [])}
Overall ADMET Score: {state.get('admet_score', 'N/A')}/1.0

Provide:
1. Executive summary (2-3 sentences)
2. Key ADMET liabilities and their clinical significance
3. Comparison to typical drug-like standards
4. Specific optimization suggestions for each liability
"""
        response = self.llm.invoke([
            SystemMessage(content="You are an expert drug metabolism and pharmacokinetics (DMPK) scientist."),
            HumanMessage(content=prompt)
        ])
        return {**state, "llm_report": response.content}

    def make_recommendation(self, state: ADMETState) -> ADMETState:
        """Final go/no-go recommendation."""
        score = state.get("admet_score", 0)
        alerts = len(state.get("structural_alerts", []))
        lipinski = state.get("lipinski_pass", False)

        if score >= 0.7 and alerts <= 1 and lipinski:
            recommendation = "ADVANCE"
        elif score >= 0.5 and alerts <= 3:
            recommendation = "OPTIMIZE"
        else:
            recommendation = "REJECT"

        return {**state, "recommendation": recommendation}

    def profile(self, smiles: str, molecule_name: str = "Unknown") -> ADMETState:
        """Profile a single molecule."""
        initial_state = ADMETState(
            smiles=smiles,
            molecule_name=molecule_name,
            absorption=None, distribution=None, metabolism=None,
            excretion=None, toxicity=None, admet_score=None,
            lipinski_pass=None, veber_pass=None,
            structural_alerts=None, llm_report=None, recommendation=None
        )
        return self.workflow.invoke(initial_state)

    def batch_profile(self, smiles_list: List[tuple]) -> pd.DataFrame:
        """Batch profile multiple molecules from [(name, smiles)] list."""
        results = []
        for name, smiles in smiles_list:
            result = self.profile(smiles, name)
            results.append({
                "name": name,
                "smiles": smiles,
                "admet_score": result.get("admet_score"),
                "recommendation": result.get("recommendation"),
                "lipinski_pass": result.get("lipinski_pass"),
                "alerts": len(result.get("structural_alerts") or [])
            })
        return pd.DataFrame(results).sort_values("admet_score", ascending=False)


if __name__ == "__main__":
    print("ADMET Oracle Orchestrator - Project 4")
    print("5 specialized agents: Absorption, Distribution, Metabolism, Excretion, Toxicity")
    print("Exposed as MCP tool for integration with drug design pipelines")
