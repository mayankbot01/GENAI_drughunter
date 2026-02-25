# MoleculeGPT: De Novo Drug Design via Fine-Tuned LLM + RLHF 🧬💊

## Vision
MoleculeGPT solves the bottleneck of generating novel, synthesizable, and bioactive molecules from scratch. Most drug discovery projects rely on existing libraries; we build new ones.

## The "Real World" Implementation
SMILES strings are not natural language — positional encoding breaks. MoleculeGPT solves this with **SELFIES tokenization** and custom vocabulary injection into the LLM tokenizer.

## Tech Stack
- **Base Models**: Fine-tuned ChemBERTa / GPT-NeoX on 2M+ ChEMBL SMILES.
- **Reinforcement Learning**: RLHF loop using a reward model for QED (drug-likeness) and SA Score (synthetic accessibility).
- **Agents**: LangChain orchestrator for scoring and iterative refinement.

## Project Structure
- `src/tokenization/`: Custom SELFIES tokenizer implementation.
- `src/rlhf/`: Reward model training and PPO loops.
- `src/generation/`: De novo molecular generation pipeline.
- `app/`: Streamlit interface for medicinal chemists.

## Resources & References
- [SELFIES: Self-Referencing Embedded Strings](https://github.com/aspuru-guzik-group/selfies)
- [RLHF for Molecular Generation](https://arxiv.org/abs/2110.14732)
- [ChEMBL Database](https://www.ebi.ac.uk/chembl/)

---
*Maintained by mayankbot01 | GENAI_drughunter Series — Project 1 of 10*
