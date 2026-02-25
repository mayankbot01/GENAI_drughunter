import os
from src.agents.tox_agent import get_tox_agent
from src.data.dataset_loader import load_tox_data
from src.utils.helpers import visualize_molecule
import argparse

def main():
    parser = argparse.ArgumentParser(description=\"ToxiSense: AI-Powered Toxicity Prediction\")
    parser.add_argument(\"--smiles\", type=str, help=\"SMILES string of the molecule to screen\")
    parser.add_argument(\"--visualize\", action=\"store_true\", help=\"Visualize the molecule\")
    args = parser.parse_args()

    # Get API Key from environment
    api_key = os.getenv(\"OPENAI_API_KEY\")
    if not api_key:
        print(\"Please set the OPENAI_API_KEY environment variable.\")
        return

    agent = get_tox_agent(api_key)

    if args.smiles:
        print(f\"Screening molecule: {args.smiles}\")
        response = agent.invoke({\"input\": f\"Analyze the toxicity of this molecule: {args.smiles}\"})
        print(f\"Agent Response: {response['output']}\")
        
        if args.visualize:
            visualize_molecule(args.smiles)
    else:
        print(\"ToxiSense CLI ready. Use --smiles to screen a molecule.\")

if __name__ == \"__main__\":
    main()
