import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
import requests
import json
import shutil
from urllib.parse import urlparse
import mimetypes
import uuid


def download_file_from_url(url, output_path):
    """
    Download a file from a URL to a specified output path.
    
    Parameters:
    - url: URL of the file to download
    - output_path: Path where the file should be saved
    
    Returns:
    - Path to the downloaded file
    """
    try:
        # Use a session with proper headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Make the request with stream=True to handle large files
        response = session.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the file
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        return output_path
    except Exception as e:
        raise Exception(f"Failed to download file from URL: {e}")


def get_file_type(file_path):
    """
    Determine if the file is a video or audio file based on its extension.
    
    Parameters:
    - file_path: Path to the file
    
    Returns:
    - 'video' or 'audio'
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    audio_extensions = ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac']
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in video_extensions:
        return 'video'
    elif file_ext in audio_extensions:
        return 'audio'
    else:
        # Try to guess from MIME type
        mime_type = mimetypes.guess_type(file_path)[0]
        if mime_type:
            if mime_type.startswith('video/'):
                return 'video'
            elif mime_type.startswith('audio/'):
                return 'audio'
    
    # Default to video if unable to determine
    return 'video'


def transcribe_and_summarize_video(video_path, temp_dir="temp", provider=None, 
                                  concurrency_level=5, chunk_length_sec=60, is_url=False):
    """
    Transcribe a video/audio file and summarize it with OpenAI GPT-4o Mini.
    Returns the summary in JSON format.
    
    Parameters:
    - video_path: Path to the video/audio file or URL
    - temp_dir: Directory for temporary files (default: "temp")
    - provider: No longer used, kept for backward compatibility
    - concurrency_level: Number of concurrent transcription processes (default: 5)
    - chunk_length_sec: Length of each audio chunk in seconds (default: 60)
    - is_url: Whether the video_path is a URL (default: False)
    
    Returns:
    - Dictionary containing the summary in JSON format
    """
    # Start timing
    start_time = time.time()
    
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Handle URL if provided
    if is_url:
        try:
            # Parse URL and get filename
            parsed_url = urlparse(video_path)
            url_path = parsed_url.path
            file_name = os.path.basename(url_path)
            
            # If no filename in URL, generate a random one
            if not file_name or '.' not in file_name:
                file_name = f"downloaded_file_{str(uuid.uuid4())[:8]}"
                # Try to determine extension from content-type
                response = requests.head(video_path)
                content_type = response.headers.get('content-type', '')
                extension = mimetypes.guess_extension(content_type)
                if extension:
                    file_name += extension
                else:
                    # Default to mp4 if can't determine type
                    file_name += '.mp4'
            
            # Download file
            download_path = os.path.join(temp_dir, file_name)
            video_path = download_file_from_url(video_path, download_path)
        except Exception as e:
            return {"error": f"Failed to download from URL: {str(e)}"}
    
    # Extract filename
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    
    # Determine if it's a video or audio file
    file_type = get_file_type(video_path)
    
    # Define paths for intermediate files
    audio_path = os.path.join(temp_dir, f"{video_name}.wav")
    chunks_dir = os.path.join(temp_dir, f"{video_name}_chunks")
    
    # STEP 1: Extract audio from video (skip this step if input is already audio)
    if file_type == 'video':
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            # Run ffmpeg command to extract audio
            command = [
                'ffmpeg', '-i', video_path, 
                '-q:a', '0', '-map', 'a', 
                '-vn', audio_path, 
                '-y'  # Overwrite if file exists
            ]
            
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
        except subprocess.CalledProcessError as e:
            return {"error": f"Audio extraction failed: {e}"}
        except Exception as e:
            return {"error": f"Audio extraction failed: {e}"}
    else:
        # For audio files, just convert to WAV format if needed
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            # Convert to WAV if not already in WAV format
            if not video_path.lower().endswith('.wav'):
                command = [
                    'ffmpeg', '-i', video_path,
                    audio_path,
                    '-y'  # Overwrite if file exists
                ]
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                # If already WAV but different path, make a copy
                if video_path != audio_path:
                    shutil.copy(video_path, audio_path)
                else:
                    # If input and output paths are the same, just use the input path
                    audio_path = video_path
                    
        except Exception as e:
            return {"error": f"Audio conversion failed: {e}"}
    
    try:
        # Create chunk directory if it doesn't exist
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Convert seconds to milliseconds
        chunk_length_ms = chunk_length_sec * 1000
        
        # Split audio into chunks
        audio_chunks = []
        for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
            # Extract chunk
            chunk = audio[start:start + chunk_length_ms]
            
            # Generate chunk file path
            chunk_file = os.path.join(chunks_dir, f"{video_name}_chunk_{i}.wav")
            
            # Export chunk
            chunk.export(chunk_file, format="wav")
            audio_chunks.append(chunk_file)
        
        if not audio_chunks:
            return {"error": "Failed to split audio into chunks"}
            
    except Exception as e:
        return {"error": f"Audio chunking failed: {e}"}
    
    # STEP 3: Transcribe audio chunks in parallel
    chunks_results = []
    time_list = []
    
    # Helper function for chunk transcription
    def process_audio_chunk(chunk_path):
        start_time = time.time()
        
        # Extract chunk_id from filename
        file_name = os.path.basename(chunk_path)
        chunk_id = os.path.splitext(file_name)[0]
        
        # Call the transcription function
        transcription_response = audio_to_text_single_call(chunk_path, temp_dir)
        
        # Extract response details
        transcript = transcription_response.get('transcript', None)
        error = transcription_response.get('error', None)
        
        chunk_result = {
            'chunk_id': chunk_id,
            'file_name': file_name,
            'error': error,
            'transcript': transcript
        }
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return chunk_result, elapsed_time
    
    # Helper function for transcription
    def audio_to_text_single_call(file_path, temp_dir):
        try:
            audio = AudioSegment.from_file(file_path)
            response_list = []

            # Handle large files (> 24MB) by further chunking
            if len(audio.raw_data) > 24 * 1024 * 1024:
                chunk_duration_ms = (24 * 1024 * 1024) / (len(audio.raw_data) / len(audio))
                chunks = make_chunks(audio, chunk_duration_ms)
                transcript = ""
                for i, chunk in enumerate(chunks):
                    chunk_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}_chunk_{i}.wav")
                    chunk.export(chunk_path, format="wav")

                    response_json, transcript_v1 = transcribe_with_openai_whisper(chunk_path)

                    transcript += transcript_v1 + " "
                    response_list.append(response_json)
                    os.remove(chunk_path)
            else:
                response_json, transcript_v1 = transcribe_with_openai_whisper(file_path)

                response_list.append(response_json)
                transcript = transcript_v1

            content = {'transcript': transcript}
            return content

        except Exception as e:
            return {'error': str(e)}

    def transcribe_with_openai_whisper(file_path):
        api_url = "https://api.openai.com/v1/audio/translations"
        
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {'error': 'OpenAI API key not found in environment variables'}, ''
        
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": "whisper-1", "prompt": "First figure out the language in audio file and then translate the audio into English", "response_format": "verbose_json"}

        with open(file_path, "rb") as file:
            files = {'file': (file_path, file, 'audio/wav')}
            response = requests.post(api_url, headers=headers, data=data, files=files)
            response_json = response.json()
            if 'error' in response_json.keys():
                return response_json, ''
            else:
                transcript = response_json['text']
            return response_json, transcript
    
    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(process_audio_chunk, chunk_path) for chunk_path in audio_chunks]
        
        for idx, future in enumerate(as_completed(futures)):
            try:
                result, elapsed_time = future.result()
                chunks_results.append(result)
                time_list.append(elapsed_time)
                
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    # STEP 4: Merge all transcripts
    try:
        # Sort chunks by their ID to maintain proper order
        sorted_chunks = sorted(chunks_results, 
                              key=lambda x: int(x['chunk_id'].split('_chunk_')[1]) 
                              if '_chunk_' in x['chunk_id'] else 0)
        
        # Combine all transcripts
        full_transcript = ' '.join([chunk['transcript'] for chunk in sorted_chunks if chunk['transcript']])
        
        if not full_transcript:
            return {"error": "No transcript was generated from any audio chunk"}
            
    except Exception as e:
        return {"error": f"Transcript merging failed: {e}"}
    
    # Helper function for time conversion
    def seconds_to_hms(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return hours, minutes, seconds
    
    # Calculate and display transcription statistics
    end_time = time.time()
    total_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(total_time)
    
    try:
        # Get API key for OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OpenAI API key not found in environment variables"}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # System and user prompts for the summarization with JSON output
        system_prompt = """
        You are a helpful assistant that summarizes legal call transcripts. Your job is to extract 
        the most important information and present it in a clear, structured JSON format. Focus only on 
        extracting key points and action items from the transcript. The transcript might contain some transcription errors.
        These calls are being made from experts on behalf of Vakilsearch.com. Correct any mis-pronunciations of the name Vakilsearch.
        """
        user_prompt = f"""
Please summarize the following legal call transcript into a structured JSON format.

Extract and organize the information into the following sections:

1. **Key Points**: Provide 3-5 bullet points summarizing the most important details discussed.
2. **Action Items**: For each action item, include:
   - **Title**: A short title for the action item.
   - **Task**: The specific task to be completed.
   - **Description**: Additional details about the task.
   - **Deadline**: The deadline, if mentioned.

Format the response as a JSON object with the structure:

{{
  "summary": {{
    "key_points": ["<Point 1>", "<Point 2>", "<Point 3>"],
    "action_items": [
      {{"title": "<Action Item Title>", "task": "<Action Item 1>", "description": "<Description>", "deadline": "<Deadline (if applicable)>"}},
      {{"title": "<Action Item Title>", "task": "<Action Item 2>", "description": "<Description>", "deadline": "<Deadline (if applicable)>"}}
    ]
  }}
}}

Return ONLY the JSON object with no additional text or explanation. Ensure it is properly formatted JSON.

Here's the transcript:
{full_transcript}
"""
        
        # API call to OpenAI GPT-4o Mini
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3  # Lower temperature for more consistent output
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                              headers=headers, 
                              data=json.dumps(data))
        response.raise_for_status()  # Raise exception for HTTP errors
        
        api_result = response.json()
        summary_text = api_result["choices"][0]["message"]["content"]
        
        # Parse the JSON summary
        try:
            summary_json = json.loads(summary_text)
            
            # Clean up temporary files
            try:
                shutil.rmtree(chunks_dir)
                os.remove(audio_path)
                
            except:
                print("Warning: Could not clean up some temporary files")
            
            return {"summary": summary_json}
            
        except json.JSONDecodeError as e:
            return {"error": "Failed to parse JSON summary", "raw_summary": summary_text}
        
    except Exception as e:
        return {"error": f"Summarization failed: {str(e)}"}

# No Flask API code - using Streamlit directly
if __name__ == "__main__":
    print("Video/Audio Transcription and Summarization Module")
    print("Use this module with Streamlit app.py for a web interface")