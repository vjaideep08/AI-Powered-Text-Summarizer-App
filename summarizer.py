from config import MODEL_CONFIG
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    """
    A text summarizer using Llama 3.2 1B model to generate 3-bullet-point summaries.
    """
    
    def __init__(self):
        """Initialize the summarizer with Llama 3.2 1B model"""
        self.model_name = MODEL_CONFIG["model_name"]
        self.max_length = MODEL_CONFIG["max_length"]
        self.temperature = MODEL_CONFIG["temperature"]
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing TextSummarizer with device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the Llama 3.2 1B model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimization for inference
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _create_prompt(self, text: str) -> str:
        original= f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful Intelligent AI assistant that creates concise summaries. Your task is to read the provided text and create exactly 3 comprehensive bullet points using a "hybrid approach" that capture the most important information. Each bullet point should be clear, concise, and informative. 

Format your response as exactly 3 bullet points, each starting with "•" and on a new line.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please summarize the following text into exactly 3 bullet points:

{text}

Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        improved_v1 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert AI assistant specializing in comprehensive text summarization. Your task is to create exactly 3 detailed bullet points that capture ALL major themes, key concepts, and important details from the provided text.

SUMMARIZATION REQUIREMENTS:
• Ensure COMPLETE COVERAGE: Each bullet point should address distinct major themes or sections from the original text
• Preserve IMPORTANT TERMINOLOGY: Use key terms, phrases, and domain-specific language from the original text
• Maintain SEMANTIC ACCURACY: Your summary should reflect the same meaning and context as the original
• Balance DETAIL with CLARITY: Each bullet point should be substantial (30-50 words) yet clear and readable
• Ensure NO MAJOR CONCEPTS are missed: Review the entire text systematically to capture all significant information

STRUCTURE GUIDELINES:
• Bullet Point 1: Cover the primary/most prominent theme or section
• Bullet Point 2: Address the secondary major theme or complementary concepts  
• Bullet Point 3: Capture remaining important details, conclusions, or supporting information

FORMAT: Respond with exactly 3 bullet points, each starting with "•" and on a new line.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please create a comprehensive 3-bullet point summary that captures ALL major themes and important details from the following text. Ensure complete coverage of the content while preserving key terminology and maintaining semantic accuracy:

{text}

Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        

        optimized_v2 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert AI assistant specializing in comprehensive yet precise text summarization. Your task is to create exactly 3 bullet points that systematically cover ALL major sections and themes from the provided text.

CORE REQUIREMENTS:
• SYSTEMATIC COVERAGE: Divide the text into 3 distinct thematic sections - ensure each bullet point covers different parts of the original text with no overlap
• PRESERVE KEY PHRASES: Use important terminology, exact phrases, and domain-specific language directly from the original text
• BALANCED DEPTH: Each bullet point should be substantive (25-40 words) but every word must add value
• SECTION MAPPING: Bullet 1 = First major theme/section, Bullet 2 = Second major theme/section, Bullet 3 = Third major theme/section or conclusion

QUALITY STANDARDS:
• HIGH PRECISION: Every word in your summary must directly relate to the original content
• COMPREHENSIVE SCOPE: Together, your 3 bullets must touch on different sentences/paragraphs from throughout the original text
• EXACT TERMINOLOGY: When possible, use the exact important phrases from the original rather than paraphrasing

FORMAT: Respond with exactly 3 bullet points, each starting with "•" and on a new line.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze the entire text below and create 3 bullet points that systematically cover different sections/themes. Ensure each bullet addresses distinct content areas and uses precise terminology from the original:

{text}

Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        simplified_final = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant that creates comprehensive summaries. Your task is to read the entire provided text carefully and create exactly 3 bullet points that together capture all the main topics and important details.

INSTRUCTIONS:
• Read through the entire text systematically
• Make sure each bullet point covers different main topics from the text
• Include important specific terms and details from the original text
• Each bullet point should be thorough but clear

Format your response as exactly 3 bullet points, each starting with "•" and on a new line.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please read the following text completely and summarize it into exactly 3 comprehensive bullet points that cover all the main topics:

{text}

Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return original
    
    def _clean_and_extract_bullets(self, generated_text: str) -> List[str]:
        """
        Extract and clean bullet points from the generated text.
        """
        # Split by the assistant token to get only the response
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            response_part = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        else:
            response_part = generated_text
        
        # Clean up the response
        response_part = response_part.strip()
        
        # Extract bullet points using regex.
        # to handle hellucinations too.
        bullet_patterns = [
            r'•\s*(.+)',  # Standard bullet
            r'-\s*(.+)',  # Dash bullet
            r'\*\s*(.+)', # Asterisk bullet
            r'\d+\.\s*(.+)' # Numbered bullet
        ]
        
        bullets = []
        lines = response_part.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in bullet_patterns:
                match = re.match(pattern, line)
                if match:
                    bullet_text = match.group(1).strip()
                    if bullet_text and len(bullet_text) > 10:  # Filter out very short bullets
                        bullets.append(bullet_text)
                    break
        
        # If we don't have exactly 3 bullets, try to extract from any line
        # Handling if three bullet points are not generated. 
        if len(bullets) != 3:
            print("ALERT !!! three bullets points are not generated!!!")
            bullets = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 10 and not line.startswith('<'):
                    bullets.append(line)
                if len(bullets) >= 3:
                    break
        
        # Ensure we have exactly 3 bullets
        if len(bullets) > 3:
            bullets = bullets[:3]
        elif len(bullets) < 3:
            # Pad with generic bullets if needed
            while len(bullets) < 3:
                bullets.append("Additional information available in the original text.")
        
        return bullets
    
    def summarize_to_bullets(self, text: str) -> List[str]:
        """
        Generate a 3-bullet-point summary of the input text.
        """
        try:
            # Validate input
            if not text or len(text.strip()) < 100:
                raise ValueError("Input text is too short for meaningful summarization")
            
            # Create prompt
            prompt = self._create_prompt(text)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,  # Limit input length
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract bullet points
            bullet_points = self._clean_and_extract_bullets(generated_text)
            
            logger.info(f"Successfully generated {len(bullet_points)} bullet points")
            return bullet_points
            
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            # Return fallback summary
            return [
                "Error occurred during summarization. Please try again.",
                "The input text could not be processed successfully.",
                "Consider checking the text format and length."
            ]
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "temperature": self.temperature,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }
