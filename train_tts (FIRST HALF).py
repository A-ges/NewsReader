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

    use_cuda = torch.cuda.is_available()
    print(f"--- Using GPU: {use_cuda} ---")
    if not use_cuda:
        print("FATAL ERROR: CUDA is not available.")
        exit()

    # --- 2. HARDCODED Paths to the Downloaded Model ---
    config_path = "C:/Users/E. Rodrigues Padrao/AppData/Local/tts/tts_models--en--ljspeech--tacotron2-DDC_ph/config.json"
    restore_path = "C:/Users/E. Rodrigues Padrao/AppData/Local/tts/tts_models--en--ljspeech--tacotron2-DDC_ph/model_file.pth"

    print(f" > Using config file at: {config_path}")
    print(f" > Using model file at: {restore_path}")

    # --- 3. Define Dataset Configuration ---
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_dev.csv",
        path=dataset_path,
    )

    # --- 4. Define Model Configuration for Fine-Tuning ---
    config = load_config(config_path)
    config.run_name = "tacotron2_ljspeech_finetune"
    config.batch_size = 8
    config.eval_batch_size = 8
    config.epochs = 50
    config.num_loader_workers = 4
    config.datasets = [dataset_config]
    config.output_path = output_path
    config.save_step = 1000

    # --- 5. Initialize Audio Processor, Model, and Trainer ---
    ap = AudioProcessor.init_from_config(config)
    model = Tacotron2.init_from_config(config)

    trainer_args = TrainerArgs(restore_path=restore_path)

    trainer = Trainer(
        trainer_args,
        config,
        output_path,
        model=model,
        train_samples=load_tts_samples(dataset_config, eval_split=False)[0],
        eval_samples=load_tts_samples(dataset_config, eval_split=True)[0],
        training_assets={"audio_processor": ap},
    )

    # --- 6. Start the Fine-Tuning ---
    print("--- Starting Coqui TTS Training ---")
    print(f" > Output folder: {output_path}")
    trainer.fit()
