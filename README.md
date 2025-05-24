# AI-Powered-Text-Summarizer-App
A lightweight AI app that allows users to input text (500 words or more) and provides a three-bullet-point summary leveraging a large language model

# 🤖 AI Text Summarizer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/)

A lightweight AI-powered web application that generates concise **3-bullet-point summaries** from long-form text using Meta's **Llama 3.2 1B Instruct** model. Perfect for quickly distilling key insights from articles, reports, documents, and other lengthy content.

## ✨ Features

- 📝 **Smart Summarization**: Generate exactly 3 comprehensive bullet points from any text
- 🎯 **Quality Control**: Enforces minimum 500-word input for meaningful summarization
- 💻 **Local Processing**: Runs entirely on your machine - no API keys or external calls
- 📱 **Responsive UI**: Clean, intuitive Streamlit interface with real-time feedback
- 📊 **Performance Tracking**: Built-in processing time monitoring and usage statistics
- 💾 **Export Capability**: Download summaries as text files
- 🔧 **Hardware Adaptive**: Automatically detects and utilizes GPU when available

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- 4GB+ RAM (8GB recommended)
- Optional: CUDA-compatible GPU for faster processing

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vjaideep08/AI-Powered-Text-Summarizer-App.git
   cd AI-Powered-Text-Summarizer-App
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

That's it! The Llama model will download automatically on first use (~2.5GB).

## 📖 Usage

1. **Input Text**: Paste or type your long-form text (minimum 500 words)
2. **Validate**: The word counter shows your progress toward the 500-word minimum
3. **Generate**: Click "🚀 Generate Summary" to create your bullet points
4. **Review**: View your 3-bullet summary with processing time
5. **Export**: Download the summary as a text file if needed

### Example Input/Output

**Input** (excerpt from a technology article):
> "Artificial intelligence has revolutionized the way we approach problem-solving across industries... [500+ words about AI impact, applications, challenges, and future prospects]"

**Output**:
• Artificial intelligence has transformed multiple industries by automating complex decision-making processes and enabling data-driven insights that were previously impossible to achieve at scale.

• Key applications span from healthcare diagnostics and financial fraud detection to autonomous vehicles and personalized recommendation systems, demonstrating AI's versatility across sectors.

• Despite significant advances, challenges remain in AI ethics, bias mitigation, and regulatory frameworks, requiring careful consideration as the technology continues to evolve rapidly.

## 🏗️ Architecture

### System Overview
```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   User      │───▶│  Streamlit   │───▶│ TextSummarizer  │
│   Input     │    │   Frontend   │    │   (Backend)     │
└─────────────┘    └──────────────┘    └─────────────────┘
                          │                       │
                          │                       ▼
                          │            ┌─────────────────┐
                          │            │ Llama 3.2 1B    │
                          │            │ Instruct Model  │
                          │            └─────────────────┘
                          │                       │
                          ▼                       ▼
                   ┌─────────────┐    ┌─────────────────┐
                   │   Results   │◀───│ Post-processing │
                   │   Display   │    │ & Validation    │
                   └─────────────┘    └─────────────────┘
```

### Core Components

- **`app.py`**: Streamlit frontend with user interface and session management
- **`summarizer.py`**: Core TextSummarizer class handling AI model operations
- **`config.py`**: Centralized configuration for model parameters and app settings
- **`requirements.txt`**: Python dependencies specification

## ⚙️ Configuration

### Model Parameters
```python
MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "max_length": 300,      # Maximum tokens for summary
    "temperature": 0.7,     # Creativity vs consistency balance
    "top_p": 0.9,          # Nucleus sampling parameter
    "top_k": 50,           # Vocabulary restriction
}
```

### Application Settings
```python
APP_CONFIG = {
    "min_words": 500,      # Minimum input word count
    "max_words": 5000,     # Recommended maximum
    "cache_model": True,   # Enable model caching
}
```

## 🛠️ Development

### Project Structure
```
ai-text-summarizer/
├── app.py                 # Streamlit frontend application
├── summarizer.py          # Core AI summarization logic
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── tests/                # Unit and integration tests
│   ├── test_summarizer.py
│   └── test_app.py
└── docs/                 # Additional documentation
    └── architecture.md
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html
```

### Code Quality
```bash
# Format code
black .

# Check style
flake8 .

# Sort imports
isort .
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

### Cloud Deployment Options

| Platform | Pros | Cons |
|----------|------|------|
| **Streamlit Cloud** | Free, Easy setup, GitHub integration | Limited resources |
| **Heroku** | Simple deployment, Good for demos | Dyno limitations |
| **AWS EC2** | Full control, GPU support | Requires setup |
| **Google Cloud Run** | Serverless, Auto-scaling | Cold starts |

## 📊 Performance

### Benchmarks
| Metric | Specification |
|--------|---------------|
| **Model Size** | ~2.5GB download |
| **Memory Usage** | 2-4GB RAM |
| **Inference Time** | 3-7 seconds (CPU), 1-3 seconds (GPU) |
| **Model Load Time** | 15-25 seconds (cached after first load) |
| **Supported Input** | Up to 8,192 tokens (~6,000 words) |

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 4GB | 8GB+ |
| **CPU** | 2 cores | 4+ cores |
| **GPU** | Not required | GTX 1060+ |
| **Storage** | 5GB free | 10GB+ |

## 🤔 Why Llama 3.2 1B?

## Overview

Llama 3.2 1B is a compact language model optimized for efficient text summarization applications. With only 1 billion parameters, it delivers competitive summarization performance while maintaining minimal computational requirements.

## Technical Specifications

| Metric | Value |
|--------|--------|
| Parameters | 1.0B |
| Model Size | ~2.5GB (FP16) |
| Context Length | 128K tokens |
| Architecture | Transformer (Llama 3.2) |
| Quantization Support | INT4, INT8, FP16 |
| Memory Requirements | 3-4GB RAM (FP16) |

## Key Features

### Performance Characteristics
- **Inference Speed**: 50-100 tokens/sec on consumer GPUs
- **Throughput**: 1000+ documents/hour on single GPU
- **Memory Efficiency**: Runs on 4GB VRAM

### Architecture Optimizations
- **Grouped Query Attention (GQA)**: Reduces memory bandwidth requirements
- **RoPE Positional Encoding**: Efficient handling of long sequences  
- **SwiGLU Activation**: Improved parameter efficiency
- **Layer Normalization**: RMSNorm for faster computation

### ✅ Advantages
- **Optimal Balance**: Great performance-to-size ratio for summarization tasks
- **Local Processing**: No API costs or rate limits
- **Fast Inference**: Quick response times suitable for interactive use
- **Instruction Following**: Excellent at structured output generation
- **Resource Efficient**: Runs on modest hardware configurations

### 📈 Model Comparison
| Model | Size | Quality | Speed | Resource Usage |
|-------|------|---------|-------|----------------|
| **Llama 3.2 1B** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GPT-3.5 Turbo | N/A | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Llama 3.2 3B | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| T5-Small | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🧪 Testing

### Test Coverage
- ✅ Unit tests for core summarization logic
- ✅ Integration tests for end-to-end pipeline
- ✅ UI component testing
- ✅ Performance benchmarking
- ✅ Error handling validation


## 🔧 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Reinstall transformers
pip uninstall transformers
pip install transformers>=4.35.0
```

**Memory Issues**
- Reduce `max_length` in `config.py`
- Use CPU-only mode: set `CUDA_VISIBLE_DEVICES=""`
- Close other applications to free RAM

**Slow Performance**
- Ensure GPU drivers are installed for CUDA support
- Check available memory with `nvidia-smi`
- Consider using smaller input texts

---

<div align="center">

**Built with ❤️ using Python, Streamlit, and Llama 3.2 1B**

[⭐ Star this repo](https://github.com/yourusername/ai-text-summarizer) • [🐛 Report Bug](https://github.com/yourusername/ai-text-summarizer/issues) • [💡 Request Feature](https://github.com/yourusername/ai-text-summarizer/issues)

</div>
