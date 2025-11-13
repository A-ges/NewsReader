"""
This is the main backend server for our Newsreader application.
It uses flask to handle requests from the frontend and orchestrates the components

Final feature set:
- Accepts a user-provided Google API Key via the frontend.
- Handles three types of input: public URL, PDF file, or Image file (JPG/JPEG).
- Extracts text from URLs, PDFs, and images (using OCR)
- Passes user options (summary length, style) to the summarization module.
- Uses our custom-trained TTS model to generate audio
- Combines audio clips into a single MP3.
- Sends the final MP3 to the user and cleans up temporary files.
"""

import os
import logging
import shutil
import json
from flask import Flask, request, send_file, jsonify, render_template, after_this_request

#Imports for File Processing
import PyPDF2
from PIL import Image
import pytesseract

#Import our custom modules
from softwareengineering import summarize_and_chunk_text, analyze_url_and_chunk
from tts_generator import initialize_synthesizer, generate_audio_clips
from audio_combiner import combine_audio_clips

#Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TEMP_AUDIO_DIR = "temp_audio_clips"

#On Windows, you must specify the path to the Tesseract executable.
#This is the default path, change it if you installed Tesseract elsewhere.
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    print("WARNING: Tesseract executable not found at the default path. OCR for images will fail.")
    print("Please ensure Tesseract is installed and the path is correctly set in app.py")

#Application setup
app = Flask(__name__, template_folder='templates')

#Global Synthesizer Initialization
logging.info("==================================================")
logging.info("Starting Newsreader Server...")
logging.info("Initializing TTS model. This may take a moment...")
synthesizer = initialize_synthesizer()
if synthesizer:
    logging.info("TTS Model ready.")
else:
    logging.error("CRITICAL: TTS Synthesizer failed to initialize. The app will not function.")
logging.info("==================================================")


def cleanup_temp_directory():
    #Removes the temporary directory for audio clips if it exists
    if os.path.exists(TEMP_AUDIO_DIR):
        try:
            shutil.rmtree(TEMP_AUDIO_DIR)
            logging.info(f"Cleaned up temporary directory: {TEMP_AUDIO_DIR}")
        except OSError as e:
            logging.error(f"Error cleaning up temporary directory {TEMP_AUDIO_DIR}: {e}")

@app.route('/')
def index():
    #Serves the main Frontend.html page
    return render_template('Frontend.html')

def extract_text_from_file(file_storage):
    #Helper function to extract text from PDF or Image file streams
    filename = file_storage.filename.lower()
    text = ""
    try:
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file_storage.stream)
            for page in pdf_reader.pages:
                text += page.extract_text() or "" # Add 'or ""' to handle empty pages gracefully
            print("Successfully extracted text from PDF.")
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(file_storage.stream)
            text = pytesseract.image_to_string(image)
            print("Successfully extracted text from Image using OCR.")
        return text
    except Exception as e:
        print(f"Error during file text extraction: {e}")
        return None

@app.route('/process', methods=['POST'])
def process_article_request():
    cleanup_temp_directory()
    logging.info("\n--- Received new request on /process ---")
    if not synthesizer:
        return jsonify({"error": "Server error: The Text-to-Speech model is not running."}), 503

    api_key = request.form.get('api_key')
    if not api_key:
        return jsonify({"error": "API Key was not provided in the request."}), 400

    try:
        options_str = request.form.get('options', '[]')
        options = json.loads(options_str)
        logging.info(f"Received options: {options}")
    except json.JSONDecodeError:
        options = ['MEDIUM_LENGTH', 'NORMAL_WORDS']
        logging.warning("Could not decode options, using defaults.")
    
    url = request.form.get('url')
    file = request.files.get('file')
    text_chunks = None

    try:
        if file and file.filename:
            logging.info(f"Processing uploaded file: {file.filename}")
            article_text = extract_text_from_file(file)
            if article_text:
                text_chunks = summarize_and_chunk_text(article_text, options, api_key)
            else:
                return jsonify({"error": "Could not extract any text from the uploaded file."}), 500
        elif url:
            logging.info(f"Processing URL: {url}")
            text_chunks = analyze_url_and_chunk(url, options, api_key)
        else:
            return jsonify({"error": "Please provide a URL or upload a file."}), 400
    except ValueError as e:
        if "Invalid API Key" in str(e):
            return jsonify({"error": "The provided Google API Key is not valid. Please check it and try again."}), 400
        return jsonify({"error": f"An input error occurred: {e}"}), 400

    if not text_chunks:
        return jsonify({"error": "Could not process the article content to generate text chunks."}), 500

    audio_clips = generate_audio_clips(synthesizer, text_chunks, output_dir=TEMP_AUDIO_DIR)
    if not audio_clips:
        return jsonify({"error": "Server error: Failed to synthesize audio."}), 500

    output_audio_file = "news_readout.mp3"
    final_audio_path = combine_audio_clips(audio_clips, output_filename=output_audio_file)
    if not final_audio_path or not os.path.exists(final_audio_path):
        return jsonify({"error": "Server error: Failed to combine audio clips."}), 500
    
    logging.info(f"Successfully created '{final_audio_path}'. Preparing to send to user.")
    
    @after_this_request
    def cleanup(response):
        #This function is called by Flask after response has been sent to user
        try:
            if os.path.exists(final_audio_path):
                os.remove(final_audio_path)
                logging.info(f"Cleaned up final audio file: {final_audio_path}")
        except Exception as error:
            logging.error(f"Error cleaning up file: {error}")
        return response

    return send_file(final_audio_path, as_attachment=True, mimetype='audio/mpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
