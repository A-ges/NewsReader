# Project description
This repository was created for our Software Engineering for CSAI course. Our task was to develop an application that assists visually impaired people in accessing information. Naturally, our group decided to create NewsReader. The goal is to build accessible summarization and text-to-speech software, specialized in news-related content.
# Newsreader application setup guide
## What is built in this instruction

Our project is a distributed system with three parts working together:

```
[Your Browser] <-> [Flask Server] <-> [RabbitMQ Queue] <-> [AI Worker] -> [Gemini + TTS]
                                                          
```

- Flask Server handles the web interface and takes your requests
- RabbitMQ manages job queue so nothing gets lost
- AI Worker does the AI related work: calling APIs and generating speech

**What you will need:**
- A PC with an NVIDIA GPU (we've used an RTX 3060 Ti to build our project)
- Updated NVIDIA drivers
- Anaconda or Miniconda

## Before Newsreader setup
### Install Erlang

Erlang is needed for RabbitMQ to function.
- Download it from: https://www.erlang.org/downloads

### Install RabbitMQ Server

RabbitMQ serves as the message queue

- Download from: https://www.rabbitmq.com/install-windows.html
- After installation, open **RabbitMQ Command Prompt** from your Start Menu and run:
  ```bash
  rabbitmq-plugins enable rabbitmq_management
  ```
- Restart the RabbitMQ in windows services
- Test it by visiting `http://localhost:15672` (login as guest)

### Install Tesseract OCR

Needed to read text from images.

- Download from: https://github.com/UB-Mannheim/tesseract/wiki

## Moving your files

Grab an external drive or use cloud storage to transfer these folders:

1. **Your entire project folder** (`NewsReader`): contains the python scripts
2. **The trained model** (`tts_training_output` folder): contains `best_model.pth` (aka the TTS model we trained)

## Setting up python

Open **Anaconda Prompt** and keep it open—you'll need it for all these steps.

### Create your environment

```bash
conda create -n tts python=3.9 -y
conda activate tts
```

### Install pytorch

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Press `y` when it asks. Don't worry if it takes some time, it is a big download.

### Install everything else

Create a file called `requirements.txt` in your project folder with this content:

```text
# Core AI & TTS
TTS==0.22.0
transformers==4.35.2

# Google Gemini (pinned for Python 3.9 compatibility)
google-generativeai==0.4.1
google-api-core==2.15.0
google-auth==2.23.4
protobuf<4

# Web server and message queue
Flask
pika

# File and audio processing
pydub
PyPDF2
Pillow
pytesseract
beautifulsoup4
requests

# Everything else
librosa
soundfile
matplotlib
pandas
scikit-learn
wget
chardet
bangla==0.0.12
```

Now it can be installed all at once:

```bash
pip install -r requirements.txt
```

### Fix the Coqui TTS Bug

There's a known issue with Python 3.9 that needs to patch manually.

Navigate to this folder in File Explorer:
```
C:\ProgramData\anaconda3\envs\tts\lib\site-packages\TTS\tts\utils\text\phonemizers
```
Replace the '__init__.py' file in this folder with the one in the "NewsReader/PATCHED __init__.py file" folder

### Check the GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

You should see: `CUDA available: True`

## Final touches

Open `ai_worker_rabbitmq.py` and make sure the tesseract path matches where you installed it:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Running everything

You need two terminal windows running at the same time.

### Terminal 1: AI Worker

```bash
conda activate tts
cd "D:\Path\To\Your\NewsReader"
python ai_worker_rabbitmq.py
```

** The worker takes 1-2 minutes to load the TTS model into memory. You'll know it's ready when you see:

```
[*] AI Worker is waiting for messages...
```

Leave this window open and running.

### Terminal 2: Web server

Open a second Anaconda Prompt:

```bash
conda activate tts
cd "D:\Path\To\Your\NewsReader"
python api_server_rabbitmq.py
```

This one starts quickly

### Now you can use NewsReader

Open your browser and go to `http://127.0.0.1:5000`

Enter your API key, paste a URL or upload a file, and click generate. Watch the AI Worker terminal to see if it shows progress.


## Fixes in case of an error

**After executing "python ai_worker_rabbitmq.py" nothing shows and the program just shows a blinking cursor**

We have this error out of nowhere since a few days before finalization, after trying so many things, I can't figure out why it happens after a restart of your PC.
a reliable fix is to do the following:
(Start a terminal with administartor permission)

```bash
conda activate tts
cd "D:\Path\To\Your\NewsReader"
pip uninstall TTS -y
pip install TTS==0.22.0 --no-cache-dir
```

**The worker is stuck on "starting up..."**

This is normal. The TTS model is huge and takes 1-2 minutes to load. If it's been more than 5 minutes, something's wrong with your PyTorch/CUDA setup.

**"TemplateNotFound: Frontend.html"**

Your Flask server can't find the HTML file. Make sure `Frontend.html` is inside a `templates` folder in your project directory.

**"Invalid API Key" or "404 model not found"**

Your Gemini API key isn't working, or you're using the wrong model. Check that:
- Your API key is valid
- `softwareengineering.py` uses `'gemini-2.5-flash'` (the older library versions need this specific model)

**TypeError about unsupported operand types on startup**

You either didn't install `bangla==0.0.12` or skipped the manual patch. Go back to "Fix the Coqui TTS Bug" and make sure you edited both lines.

**"Could not submit job to the queue"**

RabbitMQ isn't running. Open Windows Services, find RabbitMQ, and start it. You should be able to log into `http://localhost:15672` if it's working.

**CUDA shows as False**

Your GPU isn't detected. Check if:
- NVIDIA drivers are up to date
- You installed PyTorch with the conda command (not pip)
- You restarted your terminal after installation
