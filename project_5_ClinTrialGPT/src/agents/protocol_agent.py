from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    indication: str
    molecule: str
    retrieved_trials: List[dict]
    failure_risks: List[str]
    protocol_draft: str
    final_report: str

def retrieve_similar_trials(state: AgentState):
    print(f"Retrieving trials for {state['indication']}...")
    # Mock RAG retrieval
    state['retrieved_trials'] = [{"id": "NCT001", "outcome": "Success"}]
    return state

def analyze_failure_risks(state: AgentState):
    print("Analyzing historical failures...")
    state['failure_risks'] = ["Endpoint not met in Phase 2", "Safety signal"]
    return state

def generate_protocol(state: AgentState):
    print("Generating protocol with Mistral-7B...")
    state['protocol_draft'] = "Trial Protocol Draft: Indications, Endpoints..."
    return state

# Define Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_similar_trials)
workflow.add_node("analyze", analyze_failure_risks)
workflow.add_node("generate", generate_protocol)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "analyze")
workflow.add_edge("analyze", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

if __name__ == "__main__":
    inputs = {"indication": "Alzheimer's", "molecule": "Drug-X"}
    for output in app.stream(inputs):
        print(output)
