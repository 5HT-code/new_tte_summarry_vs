import streamlit as st
import json
import os
import uuid
import tempfile
from PIL import Image
import time
import shutil

# Make sure to install required packages
import subprocess
import sys

# Check if required packages are installed, install if necessary
required_packages = ['openai', 'moviepy', 'pydub']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Import the transcription function directly
from video_trans_sum import transcribe_and_summarize_video

# Set page configuration
st.set_page_config(
    page_title="Video & Audio Transcription Tool",
    page_icon="🎙️",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
    }
    .info-text {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D1FAE5;
        margin-bottom: 1rem;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FEE2E2;
        margin-bottom: 1rem;
    }
    .key-point {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #3B82F6;
        background-color: #EFF6FF;
    }
    .action-item {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        background-color: #F0FDF4;
        border: 1px solid #D1FAE5;
    }
    .stApp {
        background-color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for API keys if not present
if 'api_keys_configured' not in st.session_state:
    st.session_state.api_keys_configured = False

# Function to check API keys in Streamlit secrets
def check_api_keys():
    keys_set = False
    
    # Try to get from Streamlit secrets
    try:
        if "OPENAI_API_KEY" in st.secrets:
            keys_set = True
            st.sidebar.success("✅ OpenAI API key configured from secrets.toml")
        else:
            st.sidebar.error("❌ OpenAI API key not found in secrets.toml")
    except Exception as e:
        st.sidebar.error(f"Error accessing secrets: {e}")
    
    # Update session state
    st.session_state.api_keys_configured = keys_set
    
    return keys_set

# Title and description
st.markdown("<h1 class='main-header'>Video & Audio Transcription and Summarization</h1>", unsafe_allow_html=True)
st.markdown("""
This tool helps you transcribe and summarize audio or video content. It works with both uploaded files and URLs.
The summary extracts key points and action items to help you quickly understand the content.
""")

# Sidebar
st.sidebar.markdown("## Settings")

# API key information 
with st.sidebar.expander("API Configuration", expanded=not st.session_state.api_keys_configured):
    st.markdown("""
    ### OpenAI API Key
    The API key is configured through the `.streamlit/secrets.toml` file.
    
    Example format:
    ```toml
    OPENAI_API_KEY = "your-openai-api-key"
    ```
    
    The OpenAI API key is required for transcription and summarization.
    """)
    
    # Check API key status
    if st.button("Check API Key Status"):
        check_api_keys()

# Advanced settings
with st.sidebar.expander("Processing Settings"):
    concurrency = st.slider("Concurrency Level", 
                          min_value=1, 
                          max_value=30, 
                          value=10,
                          help="Higher values process faster but use more resources")
    
    chunk_length = st.slider("Chunk Length (seconds)", 
                           min_value=30, 
                           max_value=300, 
                           value=120,
                           help="Longer chunks may improve context but process slower")

# Request ID generation for temp directory naming
if 'request_id' not in st.session_state:
    st.session_state.request_id = str(uuid.uuid4())[:8]

custom_id = st.sidebar.text_input("Custom Request ID (optional)", 
                                 value="",
                                 help="For tracking purposes (default is auto-generated)")

if custom_id:
    st.session_state.request_id = custom_id

# Define temp directory
temp_dir = f"temp_{st.session_state.request_id}"

# Main content area
st.markdown("<h2 class='sub-header'>Input Method</h2>", unsafe_allow_html=True)
input_method = st.radio("Choose input method:", ("Upload File", "Provide URL"))

# File upload or URL input based on selection
if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload your video or audio file", 
                                   type=["mp4", "avi", "mov", "mp3", "wav", "ogg", "m4a", "flac", "aac", "mkv", "webm"])
    
    if uploaded_file:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Show a preview based on file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['mp4', 'mov', 'avi', 'webm', 'mkv']:
            st.video(uploaded_file)
        elif file_extension in ['mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac']:
            st.audio(uploaded_file)
        
        # Store in session state
        st.session_state.media_source = "file"
        st.session_state.uploaded_file = uploaded_file
        st.session_state.video_url = None
else:
    video_url = st.text_input("Enter the URL of your video or audio file")
    
    if video_url:
        st.success("URL provided")
        
        # Store in session state
        st.session_state.media_source = "url"
        st.session_state.video_url = video_url
        st.session_state.uploaded_file = None

# Ensure API keys are configured
check_api_keys()

# Process button with conditional disabling
process_disabled = (
    (not st.session_state.get('uploaded_file') and not st.session_state.get('video_url')) or 
    not st.session_state.api_keys_configured
)

if process_disabled:
    if not st.session_state.get('uploaded_file') and not st.session_state.get('video_url'):
        st.warning("Please provide a media file or URL before processing.")
    elif not st.session_state.api_keys_configured:
        st.error("The OpenAI API key is not configured in your secrets.toml file.")

if st.button("Process", type="primary", disabled=process_disabled):
    with st.spinner("Processing your media... This may take several minutes depending on the length."):
        try:
            # Create temp directory
            os.makedirs(temp_dir, exist_ok=True)
            
            if st.session_state.get('media_source') == "file" and st.session_state.get('uploaded_file'):
                # Save uploaded file to a temporary location
                uploaded_file = st.session_state.uploaded_file
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the file
                result = transcribe_and_summarize_video(
                    video_path=temp_file_path,
                    temp_dir=temp_dir,
                    concurrency_level=concurrency,
                    chunk_length_sec=chunk_length,
                    api_key=st.secrets["OPENAI_API_KEY"]
                )
                
            elif st.session_state.get('media_source') == "url" and st.session_state.get('video_url'):
                # Process from URL
                result = transcribe_and_summarize_video(
                    video_path=st.session_state.video_url,
                    temp_dir=temp_dir,
                    concurrency_level=concurrency,
                    chunk_length_sec=chunk_length,
                    is_url=True,
                    api_key=st.secrets["OPENAI_API_KEY"]
                )
            else:
                st.error("No media input found. Please upload a file or provide a URL.")
                result = {"error": "No media input provided"}
            
            # Process result
            if "error" not in result:
                # Success message
                st.markdown(f"<div class='success-box'>✅ Processing completed successfully! Request ID: {st.session_state.request_id}</div>", unsafe_allow_html=True)
                
                # Create tabs for summary and raw JSON
                tab1, tab2 = st.tabs(["Summary", "Raw JSON"])
                
                with tab1:
                    # Extract summary data
                    summary_data = result.get("summary", {})
                    
                    # Display key points
                    st.markdown("<h3>Key Points</h3>", unsafe_allow_html=True)
                    key_points = summary_data.get("key_points", [])
                    if key_points:
                        for point in key_points:
                            st.markdown(f"<div class='key-point'>• {point}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No key points found in the summary.")
                    
                    # Display action items
                    st.markdown("<h3>Action Items</h3>", unsafe_allow_html=True)
                    action_items = summary_data.get("action_items", [])
                    if action_items:
                        for item in action_items:
                            with st.container():
                                st.markdown(f"""
                                <div class='action-item'>
                                    <strong>{item.get('title', 'Untitled Action')}</strong><br>
                                    <strong>Task:</strong> {item.get('task', 'N/A')}<br>
                                    <strong>Description:</strong> {item.get('description', 'N/A')}<br>
                                    <strong>Deadline:</strong> {item.get('deadline', 'Not specified')}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No action items found in the summary.")
                
                with tab2:
                    # Display raw JSON for debugging
                    st.json(result)
                
                # Store results in session state
                st.session_state.results = result
                    
            else:
                # Error handling
                error_msg = result.get("error", "Unknown error")
                st.markdown(f"<div class='error-box'>❌ Error: {error_msg}</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"<div class='error-box'>❌ Application Error: {str(e)}</div>", unsafe_allow_html=True)
        
        finally:
            # Clean up
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                st.warning(f"Warning: Could not clean up temp directory: {e}")

# Information about supported formats
with st.expander("Supported File Formats"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Video Formats")
        st.markdown("""
        - MP4 (.mp4)
        - AVI (.avi)
        - MOV (.mov)
        - MKV (.mkv)
        - WebM (.webm)
        - FLV (.flv)
        """)
    
    with col2:
        st.markdown("### Audio Formats")
        st.markdown("""
        - MP3 (.mp3)
        - WAV (.wav)
        - OGG (.ogg)
        - M4A (.m4a)
        - FLAC (.flac)
        - AAC (.aac)
        """)

# Footer
st.markdown("---")
st.markdown("Powered by Vakilsearch © 2024")