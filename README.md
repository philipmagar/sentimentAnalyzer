# Transformer Based Product Sentiment Analyzer

This project is a Sentiment Analyzer that uses a Transformer architecture built entirely from scratch using NumPy. It analyzes product reviews and predicts whether the sentiment is positive or negative.

## The Model

The model is a custom Transformer implementation featuring:
- **Embeddings & Positional Encodings:** Maps tokens to dense vectors and injects sinusoidal positional data to preserve word order.
- **Multi-Head Self-Attention:** Uses 4 attention heads to learn contextual relationships between words in the text, regardless of their distance from each other.
- **Feed-Forward Networks & Layer Normalization:** Processes attention outputs to draw deeper conclusions and maintain mathematical stability across layers.
- **Classification Head:** Average-pools the sequence embeddings and passes them through a linear layer with softmax to output final class probabilities.

The architecture processes text sequences dynamically based on a user-defined sequence length, padding or truncating as needed, and classifies the input into `Positive` or `Negative` sentiment with an associated confidence score.

## How to Run the Project

1. **Clone the repository:**
```bash
git clone <repository-url>
cd sentimentAnalyzer
```

2. **Set up a virtual environment:**

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Mac/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit application:**
```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.
