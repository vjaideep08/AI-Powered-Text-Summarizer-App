import streamlit as st
import time
from summarizer import TextSummarizer
from config import APP_CONFIG

def remove_eot_id_from_list(strings):
    marker = "<|eot_id|>"
    cleaned_strings = []
    
    for s in strings:
        if marker in s:
            s = s.replace(marker, "")
        # else:
        #     print("Marker not found.")
        cleaned_strings.append(s)

    return cleaned_strings

st.set_page_config(
    page_title="AI-enabled Text Summarizer App",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the summarizer
@st.cache_resource
def load_summarizer():
    """Load and cache the text summarizer model"""
    return TextSummarizer()

def main():
    # Header
    st.title("🤖 AI-enabled Text Summarizer App")
    st.markdown("**Powered by Llama 3.2 1B Model**")
    st.markdown("**Created by Vobbilisetty Jayadeep**")
    st.markdown("---")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses **Llama 3.2 1B** model to generate 
        concise 3-bullet-point summaries of your text.
        
        **Features:**
        - ✅ Minimum 500 words input
        - ✅ 3-point bullet summary
        - ✅ Fast processing
        - ✅ Privacy-focused
        """)
        st.header("My Profile")

        st.markdown("""
        "Vobbilisetty Veera Venkata Jayadeep"
        jayadeepvobbilisetty@gmail.com
        """)
        
        st.header("📊 Usage count")
        if 'summary_count' not in st.session_state:
            st.session_state.summary_count = 0
        st.metric("Summaries Generated", st.session_state.summary_count)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Text")
        
        # Text input area
        input_text = st.text_area(
            "Paste your text here (minimum 500 words):",
            height=400,
            placeholder="Enter or paste your text here...",
            help="The text should be at least 500 words for optimal summarization."
        )
        
        # Word count display
        word_count = len(input_text.split()) if input_text else 0
        
        col_count, col_button = st.columns([1, 1])
        with col_count:
            if word_count < 500:
                st.error(f"Word count: {word_count}/500 (minimum required)")
            else:
                st.success(f"Word count: {word_count} ✅")
        
        with col_button:
            summarize_button = st.button(
                "🚀 Generate Summary",
                disabled=(word_count < 500),
                type="primary"
            )
    
    with col2:
        st.header("📋 Summary Output")
        
        # Summary display area
        if summarize_button and input_text and word_count >= 500:
            with st.spinner("Analyzing and summarizing your text..."):
                try:
                    # Load summarizer
                    summarizer = load_summarizer()
                    
                    # Generate summary
                    start_time = time.time()
                    summary_points = summarizer.summarize_to_bullets(input_text)
                    print(summary_points)
                    summary_points =  remove_eot_id_from_list(summary_points)
                    processing_time = time.time() - start_time
                    
                    # Display results
                    st.success("✅ Summary generated successfully!")
                    
                    # Display summary points
                    st.markdown("### Key Points:")
                    for i, point in enumerate(summary_points, 1):
                        st.markdown(f"**{i}.** {point}")
                    
                    # Display processing info
                    # st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
                    
                    # Update statistics
                    st.session_state.summary_count += 1
                    
                    # Download option
                    summary_text = "\n".join([f"• {point}" for point in summary_points])
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary_text,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")
                    st.error("Please try again or check your input text.")
        
        elif not input_text:
            st.info("👆 Please enter some text to get started!")
        elif word_count < 500:
            st.warning("⚠️ Please enter at least 500 words for summarization.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built using Llama 3.2 1B, Huggingface Transformers 🤗 and Streamlit and
        </div>
        """, 
        unsafe_allow_html=True
    )
    # st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <a href='https://www.linkedin.com/in/jayadeepvobbilisetty/'> LinkedIn Profile </a> |
            <a href='https://drive.google.com/file/d/1QCQZt0cBciM3NqP8C_w_7oQY1IV0jyoR/view?usp=sharing'> My Resume 💪 </a>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
