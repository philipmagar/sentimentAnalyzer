# Sentiment Analyzer with Transformers (Built from Scratch)

We built this for our 7th semester CSIT project. We wanted to really understand how transformers work, not just use them through some library. So we implemented the entire thing from scratch using NumPy. Yeah, it's slower than PyTorch. But we learned a ton.

The app analyzes product reviews and tells you if they're positive or negative. It's a Streamlit web interface that shows you exactly what's happening under the hood: tokenization, padding, attention mechanisms, everything.

## What It Does

Type in a review. Get back whether it's positive or negative, plus a confidence score. But here's what makes this different from other sentiment analyzers: you can see the preprocessing steps, adjust the sequence length on the fly, and watch how the model handles unknown words.

The transformer architecture uses multi-head self-attention (4 heads), sinusoidal positional encodings, and layer normalization. I coded all of that by hand in `model_utils.py`. No shortcuts.

## Why We Made This

Honestly? We were tired of treating transformers like magic black boxes. Reading the "Attention Is All You Need" paper is one thing. Actually implementing scaled dot-product attention, managing weight matrices for queries/keys/values, and debugging why your attention scores explode—that's where you really learn.

Also, we wanted something we could demo that shows both the ML side and the engineering side. The Streamlit app isn't just a prediction interface; it visualizes preprocessing, shows token-to-ID mappings, and displays model metrics. That's useful for explaining how NLP pipelines actually work.

This was our major project for 7th semester, and we wanted to build something that would actually teach us the fundamentals rather than just wrapping existing libraries.

## Getting It Running

Clone this repo:
```bash
git clone <repository-url>
cd sentimentAnalyzer
```

Set up a virtual environment (I'm on Windows, so PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you're on Mac/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```

It'll open in your browser at `localhost:8501`.

## Things We Learned (The Hard Way)

**Transformer architecture is all about data flow.** Building it from scratch taught us how data actually moves through the network—from token embeddings to positional encodings, through multi-head attention layers, into feed-forward networks, and finally to classification. When you're just using PyTorch or TensorFlow, you miss how each piece transforms the input. Implementing the attention mechanism by hand showed us why transformers are so good at capturing relationships between words, even when they're far apart in a sentence.

**Neural network data processing isn't magic, it's matrix math.** Every operation—attention scores, layer normalization, softmax—is just matrix multiplication and element-wise operations. We spent hours debugging shape mismatches (batch_size, seq_len, d_model) until we really understood how tensors flow through the network.

**Dynamic sequence length is trickier than it sounds.** Users can adjust the max token length in the sidebar (10-512 tokens). But that means the positional encoding matrix needs to regenerate, and the model has to reload. We cache the last sequence length in `st.session_state` so it only reloads when it actually changes. Otherwise, every Streamlit rerun would reload the entire model. Not fun.
## Project Structure

```
sentimentAnalyzer/
├── app.py                          # Streamlit interface
├── model_utils.py                  # Transformer implementation (all NumPy)
├── requirements.txt                # Dependencies
├── saved_hyperparameters.json      # Default hyperparams (just seq_len for now)
├── transformer_checkpoint.pkl      # Trained model weights
├── model_metrics.pkl               # Accuracy, precision, recall, F1, confusion matrix
└── screenshots/                    # UI screenshots (if you want them)
```

The checkpoint file is 11MB because it stores all the weight matrices for the embedding layer, encoder layers (queries, keys, values, output projections), feed-forward networks, and the final classification head. It's pickled Python objects, not optimized for size.

## How It Works

1. **Text preprocessing**: Lowercase everything, strip punctuation, tokenize by whitespace. I use regex to remove anything that's not alphanumeric or spaces. Simple, but it works.

2. **Encoding**: Map each token to an ID from the vocabulary. Unknown words get the `<UNK>` token ID. Then pad or truncate to the target sequence length.

3. **Positional encoding**: Add sinusoidal position embeddings so the model knows word order. This is the classic `sin/cos` approach from the original transformer paper.

4. **Transformer encoder**: Run through multiple encoder layers (each with multi-head attention + feed-forward network + layer norm). I use 4 attention heads and however many layers were in the training config.

5. **Classification**: Average-pool the sequence embeddings, then pass through a linear layer with softmax to get class probabilities.

6. **Output**: Show the predicted sentiment, confidence score, and class probabilities. Also display the preprocessing steps so you can see what the model actually "saw."

## Configuration

You can tweak the sequence length in the sidebar (default is 150 tokens). The model reloads when you change it.

The `saved_hyperparameters.json` file just stores the default:
```json
{
    "seq_len": 100
}
```
## Model Performance

If you have `model_metrics.pkl`, the sidebar shows:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix (with a heatmap)

we trained this on a product review dataset. The metrics aren't state-of-the-art, but they're decent for a from-scratch implementation. The confusion matrix usually shows that the model is better at detecting positive sentiment than negative—probably because the training data was imbalanced.


## The Team

This project was built by three CSIT students for our 7th semester major project. We split the work: one person handled the transformer implementation and training pipeline, another built the Streamlit interface and preprocessing visualization, and the third worked on model evaluation and metrics.

**Note**: This is an educational project. If you need production-grade sentiment analysis, use a library like Hugging Face Transformers. This is for learning, not for scale.
