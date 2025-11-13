# -*- coding: utf-8 -*-
"""
This script is the AI Worker for the distributed Newsreader application.
It is a long-running process that listens for jobs on a RabbitMQ queue.

Responsibilities:
1. Initialize the heavy AI models upon startup.
2. Connect to the RabbitMQ server and listen for messages on the 'task_queue'.
3. For each job received:
    a. Execute AI pipeline: extract text, summarize, chunk, synthesize audio.
    b. Update a central status file to reflect the job's progress (processing, finished, failed).
    c. Save the final MP3 result to a shared 'results' directory.
    d. Acknowledge the message to RabbitMQ to confirm successful processing.
4. Clean up all temporary files (uploads, audio clips) after each job.
"""

import os
import json
import pika
import shutil
import sys
from time import sleep

#Import AI and processing modules
from softwareengineering import summarize_and_chunk_text, analyze_url_and_chunk
from tts_generator import generate_audio_clips, initialize_synthesizer
from audio_combiner import combine_audio_clips
import PyPDF2
from PIL import Image
import pytesseract

#Worker Configuration
STATUS_FILE = 'job_status.json'
RESULTS_FOLDER = 'results'

print("AI Worker starting up, this may take a moment...")

#Model and Tool Initialization
#Heavy models are initialized once when the worker process starts.
synthesizer = initialize_synthesizer()
if not synthesizer:
    print("CRITICAL: AI Worker could not initialize TTS model. Exiting.")
    exit()

#Configure the path to the Tesseract OCR engine
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    print("WARNING: Tesseract executable not found. OCR for images will fail.")

#Helper Functions

def extract_text_from_file(file_path):
    """Extracts text content from a given PDF or image file path."""
    filename = file_path.lower()
    text = ""
    try:
        if filename.endswith('.pdf'):         
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            print("Successfully extracted text from PDF.")
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            print("Successfully extracted text from Image using OCR.")
        return text
    except Exception as e:
        print(f"Error during file text extraction: {e}")
        return None

def update_job_status(job_id, status, error=None):
    """Safely updates the central JSON status file."""
  
    with open(STATUS_FILE, 'r+') as f:
        statuses = json.load(f)
        statuses[job_id] = {"status": status}
        if error:
            statuses[job_id]['error'] = error
        f.seek(0)
        f.truncate()
        json.dump(statuses, f, indent=4)

#Main AI Processing Function

def process_job_data(job_data):
    #Contains the entire AI pipeline logic for a single job.
    job_id = job_data["job_id"]
    print(f"--- [Job {job_id}] Processing started ---")
    update_job_status(job_id, "processing")
    
    temp_audio_dir = f"temp_{job_id}"
    
    try:
        text_chunks = None
        if job_data.get("file_path"):
            article_text = extract_text_from_file(job_data["file_path"])
            if article_text:
                text_chunks = summarize_and_chunk_text(article_text, job_data["options"], job_data["api_key"])
        elif job_data.get("url"):
            text_chunks = analyze_url_and_chunk(job_data["url"], job_data["options"], job_data["api_key"])

        if not text_chunks:
            raise ValueError("Could not process input to generate text chunks.")
        
        audio_clips = generate_audio_clips(synthesizer, text_chunks, output_dir=temp_audio_dir)
        if not audio_clips:
            raise RuntimeError("Failed to synthesize any audio clips.")

        output_filename = os.path.join(RESULTS_FOLDER, f"{job_id}.mp3")
        final_audio_path = combine_audio_clips(audio_clips, output_filename=output_filename)
        if not final_audio_path:
            raise RuntimeError("Failed to combine the audio clips.")

        update_job_status(job_id, "finished")
        print(f"--- [Job {job_id}] Processing FINISHED ---")

    except Exception as e:
        print(f"!!! [Job {job_id}] FAILED. Error: {e}")
        update_job_status(job_id, "failed", error=str(e))
    finally:
        #This block runs whether the job succeeded or failed.
        if os.path.exists(temp_audio_dir):
            shutil.rmtree(temp_audio_dir)
        if job_data.get("file_path") and os.path.exists(job_data["file_path"]):
            os.remove(job_data["file_path"])


#RabbitMQ Callback Function

def callback(ch, method, properties, body):
    #This function is called by pika whenever a message is received
    print(f" [x] Received new job from queue. Delivery Tag: {method.delivery_tag}")
    job_data = json.loads(body)
    
    process_job_data(job_data)
    
    print(f" [x] Done processing job. Acknowledging message.")
    #Acknowledge the message, telling RabbitMQ it has been successfully processed
    ch.basic_ack(delivery_tag=method.delivery_tag)


#Main Worker Listening Loop

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    
    #Ensure the queue exists and is durable
    channel.queue_declare(queue='task_queue', durable=True)
    
    #This setting prevents a worker from being overwhelmed. It will only receive a new message after it has acknowledged the previous one.
    channel.basic_qos(prefetch_count=1)
    
    #Tell the channel to use our 'callback' function when a message is received
    channel.basic_consume(queue='task_queue', on_message_callback=callback)
    
    print(' [*] AI Worker is waiting for messages. To exit press CTRL+C')
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print('Interrupted by user. Shutting down.')
        channel.stop_consuming()
        connection.close()

if __name__ == '__main__':
    main()
