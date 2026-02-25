from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import torch
from ..models.gnn_model import get_model
from rdkit import Chem
import numpy as np

@tool
def predict_toxicity(smiles: str) -> str:
    \"\"\"Predicts the toxicity of a molecule given its SMILES string.\"\"\"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return \"Invalid SMILES string.\"
    
    model = get_model()
    model.eval()
    
    x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)
    adj = Chem.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    
    with torch.no_grad():
        output = model(x, edge_index, batch)
        prediction = torch.sigmoid(output).item()
    
    return f\"The predicted toxicity probability for {smiles} is {prediction:.4f}\"

def get_tox_agent(api_key):
    llm = ChatOpenAI(model=\"gpt-4-turbo\", temperature=0, openai_api_key=api_key)
    
    tools = [predict_toxicity]
    
    prompt = ChatPromptTemplate.from_messages([
        (\"system\", \"You are a toxicologist assistant using GNN models to screen drugs.\"),
        (\"human\", \"{input}\"),
        MessagesPlaceholder(variable_name=\"agent_scratchpad\"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

if __name__ == \"__main__\":
    print(\"ToxAgent module ready.\")
