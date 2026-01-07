# 🧠 Transformer-Based Product Sentiment Analyzer

A powerful web application for analyzing sentiment in product reviews using a custom transformer model built with NumPy. This Streamlit-based tool provides an interactive interface to predict whether a review expresses positive or negative sentiment, with detailed preprocessing visualization and model metrics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)


## ✨ Features

- **🤖 Transformer-Based Model**: Custom NumPy implementation of a transformer encoder for sentiment classification
- **🎯 Real-Time Analysis**: Instant sentiment prediction with confidence scores
- **📊 Model Metrics**: Display of accuracy, precision, recall, F1-score, and confusion matrix
- **🔧 Configurable Hyperparameters**: Adjustable sequence length (max tokens) via sidebar
- **🧹 Preprocessing Visualization**: See how your text is tokenized, cleaned, and encoded
- **📈 Class Probabilities**: View probability distribution across sentiment classes
- **💻 Interactive UI**: Clean, user-friendly Streamlit interface

## 🏗️ Architecture

The application consists of:

- **Transformer Encoder**: Multi-head self-attention mechanism with feed-forward networks
- **Positional Encoding**: Sinusoidal positional embeddings for sequence understanding
- **Text Preprocessing**: Lowercasing, punctuation removal, and tokenization
- **Vocabulary Mapping**: Token-to-ID conversion with `<UNK>` and `<PAD>` handling

## 📋 Requirements

- Python 3.8 or higher
- Streamlit >= 1.32.0
- NumPy >= 1.24.0
- Pandas >= 1.5.0
- Matplotlib >= 3.7.0

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd sentimentAnalyzer
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Starting the Application

Run the Streamlit app from the project directory:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

### Using the Sentiment Analyzer

1. **Enter Review Text**: Type or paste a product review in the text area
2. **Adjust Sequence Length** (Optional): Use the sidebar to modify the maximum token length (default: 150)
3. **Analyze**: Click the "🔍 Analyze Sentiment" button
4. **View Results**: 
   - See the predicted sentiment (Positive/Negative)
   - Check confidence score
   - Review preprocessing steps
   - Examine token-to-ID mappings
   - View class probabilities

### Example Reviews

**Positive Review:**
```
This product exceeded my expectations! Great quality and fast shipping. Highly recommend!
```

**Negative Review:**
```
Poor quality product. Arrived damaged and customer service was unhelpful. Very disappointed.
```

## 📁 Project Structure

```
sentimentAnalyzer/
│
├── app.py                          # Main Streamlit application
├── model_utils.py                  # Transformer model implementation and utilities
├── requirements.txt                # Python dependencies
├── saved_hyperparameters.json      # Saved model hyperparameters
├── transformer_checkpoint.pkl      # Trained model checkpoint (not included)
├── model_metrics.pkl               # Model performance metrics (optional)
├── screenshots/                    # Screenshot directory
│   ├── main_interface.png
│   ├── analysis_result.png
│   ├── model_metrics.png
│   └── preprocessing.png
└── README.md                       # This file
```

## 🔧 Model Details

The transformer model includes:

- **Vocabulary Size**: Configurable based on training data
- **Hidden Dimension (d_model)**: Model embedding dimension
- **Transformer Layers**: Number of encoder layers
- **Attention Heads**: Multi-head attention configuration
- **Sequence Length**: Maximum input token length (adjustable)

Model architecture details are displayed in the application's "Model Details" section.

## 📊 Model Metrics

The application displays comprehensive performance metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive class precision
- **Recall**: Positive class recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification performance

These metrics are loaded from `model_metrics.pkl` if available.

## 🛠️ Configuration

### Hyperparameters

You can adjust the following hyperparameter via the sidebar:

- **Sequence Length**: Maximum number of tokens to process (range: 10-512, default: 150)

The model automatically reloads when the sequence length is changed.

### Saved Configuration

Default hyperparameters are stored in `saved_hyperparameters.json`:

```json
{
    "seq_len": 100
}
```

## 🔍 How It Works

1. **Text Input**: User enters a product review
2. **Preprocessing**: 
   - Text is lowercased
   - Punctuation and special characters are removed
   - Text is tokenized into words
3. **Tokenization**: Words are mapped to vocabulary IDs
4. **Padding**: Sequence is padded/truncated to match configured length
5. **Model Inference**: Transformer processes the encoded sequence
6. **Prediction**: Softmax activation produces sentiment probabilities
7. **Result Display**: Sentiment label and confidence are shown

## ⚠️ Important Notes

- **Model Checkpoint**: The application requires `transformer_checkpoint.pkl` in the project root. Without it, predictions cannot be made.
- **Metrics File**: `model_metrics.pkl` is optional but recommended for displaying performance metrics.
- **Educational Purpose**: The NumPy-based transformer implementation is designed for educational purposes and may not be optimized for production-scale deployments.

## 🐛 Troubleshooting

### Model Not Loading

If you encounter errors loading the model:

1. Ensure `transformer_checkpoint.pkl` exists in the project root
2. Check that the checkpoint file is not corrupted
3. Verify Python version compatibility (3.8+)

### Missing Dependencies

If import errors occur:

```bash
pip install --upgrade -r requirements.txt
```

### Port Already in Use

If port 8501 is occupied:

```bash
streamlit run app.py --server.port 8502
```

## 🙏 Acknowledgments

- Streamlit for the excellent web framework
- NumPy for efficient numerical computations
- The transformer architecture inspiration from "Attention Is All You Need"
