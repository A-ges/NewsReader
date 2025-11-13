"""
This module handles the Text-to-Speech generation using a pre-trained Coqui TTS model. It initializes the synthesizer and provides a function to
convert a list of text chunks into individual audio files.
"""

import os
import torch
import logging
from TTS.utils.synthesizer import Synthesizer

#Set the name of the folder where your trained model is located.
#This should match the folder name inside your 'tts_training_output' directory.
RUN_FOLDER_NAME = "tacotron2_ljspeech_finetune-September-24-2025_11+21PM-0000000"

#Static Paths (no need to change these if your structure is consistent)
TRAINING_OUTPUT_PATH = os.path.join("tts_training_output", RUN_FOLDER_NAME)
MODEL_PATH = os.path.join(TRAINING_OUTPUT_PATH, "best_model.pth")
CONFIG_PATH = os.path.join(TRAINING_OUTPUT_PATH, "config.json")

def initialize_synthesizer():
    #Initializes and returns the Coqui TTS synthesizer + checks for necessary model files and GPU availability
    logging.info("--- Initializing TTS Synthesizer ---")
    
    #check if the model and config files exist
    if not os.path.exists(CONFIG_PATH):
        logging.error(f"Model config file not found at: {CONFIG_PATH}")
        logging.error("Please ensure the RUN_FOLDER_NAME is correct and the file exists.")
        return None
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model weights file not found at: {MODEL_PATH}")
        logging.error("Please ensure the RUN_FOLDER_NAME is correct and a 'best_model.pth' file exists.")
        return None
        
    logging.info(f"Found model config at: {CONFIG_PATH}")
    logging.info(f"Found model weights at: {MODEL_PATH}")

    #Check for GPU availability
    use_cuda = torch.cuda.is_available()
    logging.info(f"Using GPU for synthesis: {use_cuda}")

    try:
        #Initialize the Synthesizer with the paths to your trained model
        synthesizer = Synthesizer(
            tts_checkpoint=MODEL_PATH,
            tts_config_path=CONFIG_PATH,
            use_cuda=use_cuda,
        )
        logging.info("Synthesizer initialized successfully.")
        return synthesizer
    except Exception as e:
        logging.error(f"Error initializing Synthesizer: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_audio_clips(synthesizer, text_chunks, output_dir="temp_audio_clips"):
    """
    #Generates a series of .wav audio clips from a list of text chunks.
    #takes active synthesizer instance, list of strings to be synthesized into audio and the directory where temporary audio clips will be saved.
    #will return a list of file paths to the generated audio clips. Returns [] if any errors occur.
    """
    if not text_chunks:
        logging.warning("generate_audio_clips was called with no text chunks.")
        return []

    #Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f" > Generating {len(text_chunks)} audio clips in '{output_dir}' folder.")

    audio_files = []
    for i, text in enumerate(text_chunks):
        #Sanitize the text chunk to ensure it's not empty
        clean_text = text.strip()
        if not clean_text:
            continue

        file_path = os.path.join(output_dir, f"clip_{i+1:03d}.wav")
        logging.info(f"   Synthesizing chunk {i+1}: '{clean_text[:60]}...'")
        
        try:
            #Use the synthesizer's tts method to get the waveform
            wav = synthesizer.tts(clean_text)
            
            #Use the synthesizer's save_wav method to write the file
            synthesizer.save_wav(wav=wav, path=file_path)
            
            audio_files.append(file_path)
        except Exception as e:
            logging.error(f"Error synthesizing text chunk: {clean_text}")
            logging.error(f"Error details: {e}")
            #Continue to the next chunk even if one fails
            continue
            
    logging.info(f"✅ Successfully generated {len(audio_files)} audio clips.")
    return audio_files

#This block allows you to test the file directly
if __name__ == '__main__':
    #This code runs ONLY when you execute `python tts_generator.py` in your terminal
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("\n--- Running Test for tts_generator.py --")

    #initialise the synthesizer
    synth = initialize_synthesizer()

    if synth:
        #if successful, generate audio from test sentences
        print("\n--- Synthesizer Initialized. Now generating test clips... ---")
        test_sentences = [
            "Hello, this is a test of the text to speech system.",
            "The model should be able to read this sentence clearly.",
            "Let's see how it performs."
        ]
        
        generated_files = generate_audio_clips(synth, test_sentences)
        
        if generated_files:
            print("\n--- Test audio clips generated! ---")
            print(f"Please check the 'temp_audio_clips' directory for the following files:")
            for f in generated_files:
                print(f"  - {f}")
        
        else:
            print("\n--- ❌ FAILURE: Could not generate any audio clips. Check logs for errors. ---")
    else:
        print("\n--- ❌ FAILURE: Could not initialize the TTS Synthesizer. ---")
        print("Please check the configuration and error messages above.")
