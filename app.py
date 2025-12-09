import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import re
import json

from model_utils import (
    predict_sentiment,
    load_transformer_model,
    vocab,
    label_map,
    clean_text,
    encode_and_pad
)

# -----------------
# Streamlit Config
# -----------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Transformer Based Product Sentiment Analyzer")

# -----------------
# Sidebar: ONLY Sequence Length Hyperparameter
# -----------------
st.sidebar.header("⚙️ Model Hyperparameter")

seq_len_input = st.sidebar.number_input(
    "Sequence Length (Max Tokens)",
    min_value=10,
    max_value=512,
    value=150,
    step=10
)

# -----------------
# Track seq_len for reload
# -----------------
if "last_seq_len" not in st.session_state:
    st.session_state.last_seq_len = None

current_seq_len = seq_len_input

# -----------------
# Conditional Model Reload (ONLY seq_len)
# -----------------
if current_seq_len != st.session_state.last_seq_len:
    with st.spinner("Reloading model with updated sequence length..."):
        st.session_state.model, st.session_state.vocab, st.session_state.label_map = load_transformer_model(
            override_params={"seq_len": current_seq_len}
        )
        st.session_state.last_seq_len = current_seq_len

model = st.session_state.model
vocab = st.session_state.vocab
label_map = st.session_state.label_map

# -----------------
# ✅ MODEL DETAILS (ADDED HERE)
# -----------------
st.subheader("🧠 Model Details")

try:
    model_details = {
        "Vocabulary Size": len(vocab),
        "Hidden Dimension (d_model)": getattr(model, "d_model", "Not Available"),
        "Number of Transformer Layers": getattr(model, "num_layers", "Not Available"),
        "Number of Attention Heads": getattr(model, "num_heads", "Not Available"),
        "Active Sequence Length": current_seq_len
    }

    st.json(model_details)

except Exception as e:
    st.error(f"Failed to load model details: {e}")

# -----------------
# Display Active Hyperparameter
# -----------------
st.subheader("🔧 Active Hyperparameter")
st.json({"seq_len": current_seq_len})

# -----------------
# Sidebar: Model Metrics
# -----------------
st.sidebar.header("📊 Model Metrics")
metrics_path = "model_metrics.pkl"

if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)

        st.sidebar.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        st.sidebar.write(cm)

        st.sidebar.subheader("Metrics")
        st.sidebar.write(f"Accuracy : {metrics['accuracy']:.4f}")
        st.sidebar.write(f"Precision: {metrics['precision']:.4f}")
        st.sidebar.write(f"Recall   : {metrics['recall']:.4f}")
        st.sidebar.write(f"F1-score : {metrics['f1_score']:.4f}")

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(cm, cmap='Blues')
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Positive", "Negative"])
        ax.set_yticklabels(["Positive", "Negative"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

        st.sidebar.pyplot(fig)

    except Exception as e:
        st.sidebar.error(f"Failed to load metrics: {e}")
else:
    st.sidebar.warning("Metrics file not found.")

# -----------------
# Review Input
# -----------------
text_input = st.text_area(
    "Enter review text",
    height=150,
    placeholder="Type a review here..."
)

# -----------------
# Analyze Button & Cleaning
# -----------------
if st.button("🔍 Analyze Sentiment"):
    if not text_input.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Analyzing..."):

        def safe_clean_text(text):
            text = text.lower()
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            return text.split()

        cleaned = safe_clean_text(text_input.strip())

        result = predict_sentiment(
            text_input.strip(),
            model,
            vocab,
            label_map
        )

    # -----------------
    # Preprocessing Display
    # -----------------
    st.subheader("🧹 Preprocessing")
    st.code(" ".join(cleaned))

    unk_id = vocab.get("<UNK>", -1)
    token_ids = [vocab.get(tok, unk_id) for tok in cleaned]

    df_map = pd.DataFrame({
        "Token": cleaned,
        "Token ID": token_ids,
        "In Vocabulary": ["YES" if t in vocab else "NO" for t in cleaned]
    })

    st.dataframe(df_map, use_container_width=True)

    # -----------------
    # Padding with ONLY seq_len
    # -----------------
    padded = encode_and_pad(cleaned, vocab, current_seq_len)

    st.write("Padded Sequence:")
    st.code(padded)

    # -----------------
    # Final Prediction Result
    # -----------------
    st.subheader("📌 Result")

    sentiment = result["sentiment"]
    confidence = result["confidence"] * 100

    if sentiment == "Positive":
        st.success(f"Sentiment: {sentiment} 👍")
    else:
        st.error(f"Sentiment: {sentiment} 👎")

    st.metric("Confidence", f"{confidence:.2f}%")

    # -----------------
    # Class Probabilities
    # -----------------
    st.subheader("📈 Class Probabilities")
    for label, prob in result["probabilities"].items():
        st.write(f"{label}: {prob * 100:.2f}%")
