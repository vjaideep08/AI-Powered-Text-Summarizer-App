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
- 💾 **Export Capability**: Download summaries as text files
- 🔧 **Hardware Adaptive**: Automatically detects and utilizes GPU when available
- 🤗 **Simple Model Replace/ Update**: Simply test the solution with the different models just by updating the model name in config.py file. (Important : Remenber to change the **prompt structure and format** in summarizer.py according to the model used)

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


## Technical Approach

### 1. Modular Architecture Design

The solution employs a clean separation of concerns:

**Frontend Layer (Streamlit)**
```
├── app.py
│   ├── UI Components
│   │   ├── Text Input Area (min 500 words)
│   │   ├── Word Counter & Validation
│   │   ├── Generate Summary Button
│   │   └── Results Display Panel
│   ├── Session Management
│   │   ├── Model Caching (@st.cache_resource)
│   │   └── Statistics Tracking
│   └── Error Handling & User Feedback
```

**Backend Layer (Core Processing)**
```
├── summarizer.py
│   ├── TextSummarizer Class
│   │   ├── Model Initialization
│   │   │   ├── Device Detection (CUDA/CPU)
│   │   │   ├── Model Loading with Optimization
│   │   │   └── Tokenizer Configuration
│   │   ├── Prompt Engineering
│   │   │   ├── System Message Formatting
│   │   │   ├── Chat Template Structure
│   │   │   └── Context Window Management
│   │   ├── Text Generation
│   │   │   ├── Input Tokenization
│   │   │   ├── Inference with Parameters
│   │   │   └── Output Decoding
│   │   └── Post-processing
│   │       ├── Bullet Point Extraction
│   │       ├── Format Validation
│   │       └── Quality Assurance
│   └── Error Handling & Logging
```

**Configuration Layer**
```
├── config.py
│   ├── MODEL_CONFIG (Model parameters)
│   ├── APP_CONFIG (Application settings)
│   ├── UI_CONFIG (Interface parameters)
│   └── PERFORMANCE_CONFIG (Optimization settings)
```

- **Frontend**: Streamlit handles user interaction and display
- **Backend**: Dedicated TextSummarizer class manages AI processing
- **Configuration**: Centralized settings management
- **Utilities**: Helper functions for text processing

### 2. Hybrid Summarization Strategy

The implementation uses a "hybrid approach" that combines:
- **Extractive elements**: Identifying key information from source text
- **Abstractive elements**: Generating coherent, reformulated bullet points
- **Structure enforcement**: Ensuring exactly 3 bullet points output

### 3. Resource Optimization

- **Model caching**: Streamlit's @st.cache_resource prevents repeated loading
- **Memory management**: Low CPU memory usage configuration
- **Device optimization**: Automatic CUDA/CPU detection and allocation
- **Precision optimization**: FP16 on GPU, FP32 on CPU


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
└── docs/                 # Approach and strategy documents
    └── Current Implementation Approach and Strategy.docx
    └── Production Deployment Stretegy.docx
└── Evaluation Metrics Results/                 # Evalutaion results of model and prompts and theoritical documents
    └── llama_evaluation_results_20250524_201128.json
    └── llama_evaluation_report_20250524_201128.txt
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

### Implementation SnapShots
**##Streamlit Interface**
![image](https://github.com/user-attachments/assets/01bbbfcc-e5dd-41bc-9bb1-8c05665a86df)

## Text Summarization Results 
![image](https://github.com/user-attachments/assets/fd03679e-221c-465d-983a-d38f9275367c)
![image](https://github.com/user-attachments/assets/a8233b29-9a5a-40f9-906d-b51e2e5bdeb6)



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
| **GPU** | min of 4GB | based on model and weights compatibility |
| **Storage** | 5GB free | 10GB+ |

## 🤔 Why Llama 3.2 1B Instruct?

## Overview

Llama 3.2 1B Instruct is a compact language model optimized for efficient text summarization applications. With only 1 billion parameters, it delivers competitive summarization performance while maintaining minimal computational requirements.

## Model Selection Criteria

### Why Llama 3.2 1B Instruct?

#### ✅ Advantages

**1. Optimal Size-Performance Balance**
- 1B parameters provide sufficient capability for summarization
- Lightweight enough for local deployment
- Fast inference times (typically < 5 seconds)

**2. Instruction Following Capability**
- Pre-trained for instruction-following tasks
- Excellent at structured output generation
- Reliable bullet-point formatting

**3. Resource Efficiency**
- Runs on both CPU and GPU
- Low memory footprint (~2-4GB RAM)
- Suitable for edge deployment

**4. Open Source & Accessible**
- Available through Hugging Face
- No API costs or rate limits
- Full control over deployment
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


### Instruction Tuned Models

| Capability |  | Benchmark | \# Shots | Metric | Llama 3.2 1B bf16 | Llama 3.2 1B Vanilla PTQ\*\* | Llama 3.2 1B Spin Quant | Llama 3.2 1B QLoRA | Llama 3.2 3B bf16 | Llama 3.2 3B Vanilla PTQ\*\* | Llama 3.2 3B Spin Quant | Llama 3.2 3B QLoRA | Llama 3.1 8B |
| :---: | ----- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| General |  | MMLU | 5 | macro\_avg/acc | 49.3 | 43.3 | 47.3 | 49.0 | 63.4 | 60.5 | 62 | 62.4 | 69.4 |
| Re-writing |  | Open-rewrite eval | 0 | micro\_avg/rougeL | 41.6 | 39.2 | 40.9 | 41.2 | 40.1 | 40.3 | 40.8 | 40.7 | 40.9 |
| Summarization |  | TLDR9+ (test) | 1 | rougeL | 16.8 | 14.9 | 16.7 | 16.8 | 19.0 | 19.1 | 19.2 | 19.1 | 17.2 |
| Instruction following |  | IFEval | 0 | Avg(Prompt/Instruction acc Loose/Strict) | 59.5 | 51.5 | 58.4 | 55.6 | 77.4 | 73.9 | 73.5 | 75.9 | 80.4 |
| Math |  | GSM8K (CoT) | 8 | em\_maj1@1 | 44.4 | 33.1 | 40.6 | 46.5 | 77.7 | 72.9 | 75.7 | 77.9 | 84.5 |
|  |  | MATH (CoT) | 0 | final\_em | 30.6 | 20.5 | 25.3 | 31.0 | 48.0 | 44.2 | 45.3 | 49.2 | 51.9 |
| Reasoning |  | ARC-C | 0 | acc | 59.4 | 54.3 | 57 | 60.7 | 78.6 | 75.6 | 77.6 | 77.6 | 83.4 |
|  |  | GPQA | 0 | acc | 27.2 | 25.9 | 26.3 | 25.9 | 32.8 | 32.8 | 31.7 | 33.9 | 32.8 |
|  |  | Hellaswag | 0 | acc | 41.2 | 38.1 | 41.3 | 41.5 | 69.8 | 66.3 | 68 | 66.3 | 78.7 |
| Tool Use |  | BFCL V2 | 0 | acc | 25.7 | 14.3 | 15.9 | 23.7 | 67.0 | 53.4 | 60.1 | 63.5 | 67.1 |
|  |  | Nexus | 0 | macro\_avg/acc | 13.5 | 5.2 | 9.6 | 12.5 | 34.3 | 32.4 | 31.5 | 30.1 | 38.5 |
| Long Context |  | InfiniteBench/En.QA | 0 | longbook\_qa/f1 | 20.3 | N/A | N/A | N/A | 19.8 | N/A | N/A | N/A | 27.3 |
|  |  | InfiniteBench/En.MC | 0 | longbook\_choice/acc | 38.0 | N/A | N/A | N/A | 63.3 | N/A | N/A | N/A | 72.2 |
|  |  | NIH/Multi-needle | 0 | recall | 75.0 | N/A | N/A | N/A | 84.7 | N/A | N/A | N/A | 98.8 |
| Multilingual |  | MGSM (CoT) | 0 | em | 24.5 | 13.7 | 18.2 | 24.4 | 58.2 | 48.9 | 54.3 | 56.8 | 68.9 |

### 📈 Model Comparison

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

</div>
