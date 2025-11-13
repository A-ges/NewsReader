"""
This module processes text content with the Google Gemini API.
It is designed to be called by the main Flask application.

Functions:
- Summarizes provided text based on user-selected options from frontend
- Splits the final text into unit-test defined groups for our own TTS engine
- Accepts API key on a per-request basis for security and flexibility
"""

import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import json
import re


#The API key is now configured dynamically inside the functions when they are called
#This allows the user to provide their key via the frontend

def chunk_text_with_gemini(text_to_chunk, model):
    """
    Uses the Gemini API to split text into natural-sounding chunks for TTS.
    This function is a utility and does not change.
    """
    print("Sending text to Gemini for intelligent chunking...")
    prompt = f"""
    You are a text pre-processor for a high-quality Text-to-Speech (TTS) system.
    Your task is to break the following text into short, natural-sounding phrases suitable for TTS. Keep in mind that the TTS is made for English speech, so it would be best if you would translate the text int english, with the first sentence saying which language it was translated from.
    Follow these rules strictly:
    1. Each phrase must be between 6 and 12 words long.
    2. Split the text at natural pauses like commas or the end of chunks.
    3. Do not lose any words from the original text.
    4. Your output must be a valid JSON object with a single key "chunks" which holds an array of the text strings.
    5. Keep the output simply to what is instructed, so no "Sure, here is your output"-esque things sprinkled in between. 
    Here is the text to process:
    ---
    {text_to_chunk}
    ---
    Provide only the raw JSON output.
    """
    try:
        response = model.generate_content(prompt)
        cleaned_response_text = re.sub(r'```json\s*|\s*```', '', response.text.strip())
        data = json.loads(cleaned_response_text)
        chunks = data.get("chunks")
        if chunks:
            print("Successfully received chunks from Gemini.")
            return chunks
        else:
            print("Error: JSON was valid but did not contain the 'chunks' key.")
            return None
    except Exception as e:
        print(f"An unexpected error occurred during Gemini chunking: {e}")
        return None

def summarize_and_chunk_text(article_text, options, api_key):
    #Takes raw text, summarizes it based on options, and chunks it.
    #It now configures the API key for this specific request.
    try:
        #Configure the genai library with the user-provided key
        genai.configure(api_key=api_key)
        
        #Initialize the generative model
        model = genai.GenerativeModel('gemini-2.5-flash')

        #Dynamically build the prompt based on frontend options
        if 'FULL_TEXT' in options:
            summary_text = article_text  #Skip summarization entirely to save some time
        else:
            prompt_instructions = "Summarize this text. " #customize the prompt
            if 'SHORT_LENGTH' in options:
                prompt_instructions += "The summary should be short and concise, about 2-3 sentences. "
            elif 'LONG_LENGTH' in options:
                prompt_instructions += "The summary should be long and detailed, covering all key points. "
            
            if 'EASIER_WORDS' in options:
                prompt_instructions += "Use simple, easy-to-understand language. "

            print(f"Requesting summary from Gemini with instructions: '{prompt_instructions}'")
            summary_prompt = [prompt_instructions, article_text]
            summary_response = model.generate_content(summary_prompt)
            summary_text = summary_response.text

        print("--- Generated Summary ---")
        print(summary_text)
        print("-------------------------")

        #Chunk the final text (either summary or full text)
        sentence_chunks = chunk_text_with_gemini(summary_text, model)
        return sentence_chunks

    except Exception as e:
        #Check for a common API key error and raise a specific, catchable error
        if "API key not valid" in str(e):
            print("An invalid Google API Key was provided.")
            raise ValueError("Invalid API Key") from e
        
        print(f"An unexpected error occurred during Gemini processing: {e}")
        return None
def analyze_url_and_chunk(article_url, options, api_key):
    #To fetch content from a URL and then pass all arguments (including the api_key) to the core text processing function.
    print(f"Starting analysis for URL: {article_url}")
    try:
        response = requests.get(article_url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        article_text = ""
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            article_text += p.get_text() + "\n"
        
        if not article_text.strip():
            print("Error: Could not extract any paragraph text from the URL.")
            return None
        
        #call the core function with the extracted text and the API key
        return summarize_and_chunk_text(article_text, options, api_key)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching article from URL: {e}")
        return None
