# Sentiment Analyzer (Transformer-based)

A compact Streamlit web app for running sentiment analysis on short product reviews using a small transformer-style model.

## Features
- Interactive web UI powered by Streamlit (`app.py`) to enter/paste a review and get a sentiment prediction.
- Lightweight NumPy-based transformer implementation and helper utilities in `model_utils.py`.
- Configurable active sequence length (max tokens) from the Streamlit sidebar.

## Requirements
Python 3.8+ and the dependencies listed in `req.txt`.

## Installation
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
pip install -r req.txt
```

## How to run
Start the Streamlit app from the project directory:

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501) to use the web UI.

## Files of note
- `app.py` — Streamlit UI, loads model and presents inputs/results.
- `model_utils.py` — Lightweight transformer code, tokenizer helpers, model loader, and predict helper.
- `transformer_checkpoint.pkl` — (Not included) Binary checkpoint expected by `load_transformer_model`.
- `model_metrics.pkl` — (Optional) Metrics used to show confusion matrix and performance in the sidebar.
- `saved_hyperparameters.json` — Saved defaults (e.g., `seq_len`).

If you don't have `transformer_checkpoint.pkl` in the project root, `app.py` will fall back and the UI may not be able to make predictions. Place your trained checkpoint file at the path `transformer_checkpoint.pkl`.

## Notes / Tips
- The app's sidebar allows changing the `Sequence Length (Max Tokens)` — the model will reload when this value changes.
- `model_utils.py` contains a simple, NumPy-only Transformer implementation intended for small educational checkpoints — it's not optimized for production use.

## License
This repository doesn't include a license file.