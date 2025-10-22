import wget, tarfile, os
import pandas as pd
import time
from sklearn.model_selection import train_test_split

print("--- Starting LJSpeech Download & Extraction ---")

# Ensure you are in the correct directory, or specify a path
if not os.path.exists("LJSpeech-1.1"):
    # Download dataset
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    print(f"Downloading LJSpeech from {url}...")
    start_time = time.time()
    wget.download(url, bar=wget.bar_adaptive)
    print(f"\nDownload complete in {time.time() - start_time:.2f} seconds.")

    # Extract
    archive_name = "LJSpeech-1.1.tar.bz2"
    print(f"Extracting {archive_name}...")
    start_time = time.time()
    with tarfile.open(archive_name, "r:bz2") as tar:
        tar.extractall()
    print(f"Extraction complete in {time.time() - start_time:.2f} seconds.")
    # Clean up the archive to save space
    os.remove(archive_name)
    print(f"Removed archive: {archive_name}")
else:
    print("LJSpeech-1.1 directory already exists. Skipping download and extraction.")

# Data exploration
metadata = pd.read_csv("LJSpeech-1.1/metadata.csv", sep="|",
                      names=["id", "transcription", "normalized"])

print("\n--- LJSpeech Dataset Overview ---")
print(f"Volume: {len(metadata)} clips, estimated ~24 hours")
print(f"Variety: Single speaker, English, non-fiction")
print(f"Velocity: Static dataset")
print(f"Veracity: LibriVox project, public domain")
print("\nFirst 5 rows of metadata:")
print(metadata.head())
print(f"\nExample clip ID: {metadata['id'].iloc[0]}. Corresponding audio file: LJSpeech-1.1/wavs/{metadata['id'].iloc[0]}.wav")

print("--- LJSpeech Download & Exploration Complete ---")


# --- Data Preprocessing Section ---
print("\n--- Starting Data Preprocessing (Splitting & Cleaning) ---")

# Create train/dev/test splits (80/10/10)
train_data, temp_data = train_test_split(metadata, test_size=0.2, random_state=42)
dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Original Split Sizes: Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

# Clean data - remove very short/long clips for better training stability
initial_train_len = len(train_data)
train_data_clean = train_data[
    (train_data['normalized'].str.len() > 10) &
    (train_data['normalized'].str.len() < 200)
]
print(f"Train data cleaned: Removed {initial_train_len - len(train_data_clean)} clips (length >10 and <200 chars).")

# Also clean dev and test data
initial_dev_len = len(dev_data)
dev_data_clean = dev_data[
    (dev_data['normalized'].str.len() > 10) &
    (dev_data['normalized'].str.len() < 200)
]
print(f"Dev data cleaned: Removed {initial_dev_len - len(dev_data_clean)} clips.")

initial_test_len = len(test_data)
test_data_clean = test_data[
    (test_data['normalized'].str.len() > 10) &
    (test_data['normalized'].str.len() < 200)
]
print(f"Test data cleaned: Removed {initial_test_len - len(test_data_clean)} clips.")

print(f"Final Split Sizes after cleaning: Train: {len(train_data_clean)}, Dev: {len(dev_data_clean)}, Test: {len(test_data_clean)}")

# Save the split metadata files for Coqui TTS
train_data_clean.to_csv("LJSpeech-1.1/metadata_train.csv", sep="|", index=False, header=False)
dev_data_clean.to_csv("LJSpeech-1.1/metadata_dev.csv", sep="|", index=False, header=False)
test_data_clean.to_csv("LJSpeech-1.1/metadata_test.csv", sep="|", index=False, header=False)

print("\nMetadata split files (metadata_train.csv, metadata_dev.csv, metadata_test.csv) saved to LJSpeech-1.1/ directory.")
print("--- Data Preprocessing Complete ---")
