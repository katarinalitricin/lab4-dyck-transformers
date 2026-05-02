# Transformers and Dyck Languages

Lab project for studying error detection, correction, OOD generalisation, attention, and probing in small Transformer encoders trained on Dyck languages.

## Structure

- `src/data.py`: Dyck data generation, corruptions, tokenisation
- `src/model.py`: Transformer encoder models
- `src/train.py`: training scripts
- `src/evaluate.py`: evaluation and plots
- `src/baseline.py`: deterministic stack baseline
- `notebooks/`: exploration and analysis
- `report/`: report notes and final answers

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

