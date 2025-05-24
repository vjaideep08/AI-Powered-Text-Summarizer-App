# Model Configuration
MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "max_length": 300,  # Maximum tokens for summary generation
    "temperature": 0.7,  # Sampling temperature for generation
    "top_p": 0.9,       # Nucleus sampling parameter
    "top_k": 50,        # Top-k sampling parameter
}

# Application Configuration
APP_CONFIG = {
    "title": "AI Text Summarizer",
    "description": "Generate concise 3-bullet-point summaries using Llama 3.2 1B",
    "min_words": 500,   # Minimum word count for input text
    "max_words": 5000,  # Maximum recommended word count
    "page_icon": "📝",
    "layout": "wide"
}

# UI Configuration
UI_CONFIG = {
    "input_height": 400,
    "placeholder_text": "Enter or paste your long-form text here...",
    "help_text": "The text should be at least 500 words for optimal summarization.",
    "button_text": "🚀 Generate Summary",
    "success_message": "✅ Summary generated successfully!",
    "error_message": "❌ Error generating summary"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": "summarizer.log"
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "cache_model": True,
    "low_memory_mode": True,
    "batch_size": 1,
    "max_input_tokens": 2048
}
