import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import pika
from werkzeug.utils import secure_filename

#Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
STATUS_FILE = 'job_status.json'
FEEDBACK_FILE = 'feedback.txt' #To be written file for storing user feedback
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

#App setup
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_rabbitmq_connection():
    return pika.BlockingConnection(pika.ConnectionParameters('localhost'))

if not os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, 'w') as f:
        json.dump({}, f)

#Flask Routes

@app.route('/')
def index():
    return render_template('Frontend.html')

@app.route('/submit-job', methods=['POST'])
def submit_job():
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "api_key": request.form.get('api_key'),
        "options": json.loads(request.form.get('options', '[]')),
        "url": request.form.get('url'),
        "file_path": None
    }
    if not job_data["api_key"]:
        return jsonify({"error": "API Key is required."}), 400
    file = request.files.get('file')
    if file and file.filename:
        filename = secure_filename(f"{job_id}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        job_data["file_path"] = file_path
    elif not job_data["url"]:
        return jsonify({"error": "A URL or a file is required."}), 400
    try:
        connection = get_rabbitmq_connection()
        channel = connection.channel()
        channel.queue_declare(queue='task_queue', durable=True)
        channel.basic_publish(
            exchange='',
            routing_key='task_queue',
            body=json.dumps(job_data),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        connection.close()
        print(f"Job {job_id} published to RabbitMQ.")
    except Exception as e:
        print(f"Error publishing to RabbitMQ: {e}")
        return jsonify({"error": "Could not submit job to the queue."}), 500
    with open(STATUS_FILE, 'r+') as f:
        statuses = json.load(f)
        statuses[job_id] = {"status": "queued"}
        f.seek(0)
        json.dump(statuses, f, indent=4)
    return jsonify({"job_id": job_id})

@app.route('/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    with open(STATUS_FILE, 'r') as f:
        statuses = json.load(f)
    job_status = statuses.get(job_id)
    if job_status:
        if job_status.get("status") == "finished":
            job_status["result_url"] = url_for('get_result_file', filename=f"{job_id}.mp3")
        return jsonify(job_status)
    else:
        return jsonify({"status": "not_found"}), 404

@app.route('/results/<filename>', methods=['GET'])
def get_result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)


#Handling user feedback 
@app.route('/review', methods=['POST'])
def handle_review():
#Accepts user feedback submitted from the frontend and saves it to a text file.
    try:
        data = request.get_json()
        feedback_text = data.get('review')
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())

        if not feedback_text:
            return jsonify({"error": "Review text cannot be empty."}), 400

        #Format the entry to be saved
        log_entry = f"Timestamp: {timestamp}\nFeedback: \"{feedback_text}\"\n--------------------------------\n"

        #Append the feedback to our file
        with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"Received and saved new feedback.")
        return jsonify({"status": "success", "message": "Feedback received."}), 200

    except Exception as e:
        print(f"Error saving feedback: {e}")
        return jsonify({"error": "Server error while saving feedback."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
