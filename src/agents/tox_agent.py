from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import torch
import numpy as np
from rdkit import Chem

# Corrected imports for package structure consistency
try:
    from src.models.gnn_model import get_model
    from src.utils.helpers import get_atom_features
except ImportError:
    from ..models.gnn_model import get_model
    from ..utils.helpers import get_atom_features

@tool
def predict_toxicity(smiles: str) -> str:
    """Predicts the toxicity of a molecule given its SMILES string using a GNN model."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string."
    
    # Initialize model with 78 features (consistent with featurizer)
    model = get_model(num_features=78)
    model.eval()
    
    # Generate node features (78 dimensions per atom)
    node_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.array(node_feats), dtype=torch.float)
    
    # Generate adjacency matrix for edge index
    adj = Chem.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    
    # Batch tensor for global pooling (all atoms in one molecule)
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    
    with torch.no_grad():
        output = model(x, edge_index, batch)
        prediction = torch.sigmoid(output).item()
    
    return f"The predicted toxicity probability for {smiles} is {prediction:.4f}"

def get_tox_agent(api_key):
    """Initializes the LangChain agent for toxicity screening."""
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=api_key)
    
    tools = [predict_toxicity]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a toxicologist assistant using GNN models to screen drugs. You provide accurate toxicity predictions based on molecular structures."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

if __name__ == "__main__":
    print("ToxAgent module ready and consistent with advanced featurization.")
