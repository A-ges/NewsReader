import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.config import load_config
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.utils.audio import AudioProcessor

# This is the standard protection for Windows multiprocessing
if __name__ == '__main__':
    # --- 1. Define Paths and Check for GPU ---
    output_path = "tts_training_output"
    dataset_path = "LJSpeech-1.1/"
    
    run_folder_name = "tacotron2_ljspeech_finetune-September-24-2025_11+21PM-0000000"
    
    use_cuda = torch.cuda.is_available()
    print(f"--- Using GPU: {use_cuda} ---")
    if not use_cuda:
        print("FATAL ERROR: CUDA is not available.")
        exit()

    # --- 2. Define path for resuming training ---
    # The 'continue_path' tells the trainer to load the last checkpoint from this folder
    continue_path = os.path.join(output_path, run_folder_name)
    print(f" > Attempting to resume training from: {continue_path}")
    
    # The config path is needed to initialize the model structure
    config_path = os.path.join(continue_path, "config.json")
    print(f" > Using config file at: {config_path}")

    # --- 3. Define Dataset Configuration ---
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_dev.csv",
        path=dataset_path,
    )

    # --- 4. Load the Model Configuration ---
    config = load_config(config_path)

    # --- 5. Initialize Audio Processor, Model, and Trainer ---
    ap = AudioProcessor.init_from_config(config)
    model = Tacotron2.init_from_config(config)

    # Use 'continue_path' to resume the previous training run
    trainer_args = TrainerArgs(continue_path=continue_path)

    trainer = Trainer(
        trainer_args,
        config,
        output_path, # This should be the parent folder
        model=model,
        train_samples=load_tts_samples(dataset_config, eval_split=False)[0],
        eval_samples=load_tts_samples(dataset_config, eval_split=True)[0],
        training_assets={"audio_processor": ap},
    )

    # --- 6. Resume the Fine-Tuning ---
    print("--- Resuming Coqui TTS Training ---")
    trainer.fit()
