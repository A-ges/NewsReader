from TTS.api import TTS
import torch

print("Downloading and caching the pre-trained model")
print()

#This line will connect to the internet, download the required model files and save them in the correct location for future use.
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC_ph", progress_bar=True, gpu=torch.cuda.is_available())

print("Model download complete")
print("You can now run the train_tts.py script")

