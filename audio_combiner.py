"""
This module is responsible for combining multiple audio files into a single file.
It uses the pydub library to concatenate audio segments and exports them
into a final MP3 format. It also handles the cleanup of temporary files.
"""

import os
import logging
from pydub import AudioSegment, exceptions as pydub_exceptions

def combine_audio_clips(clip_paths, output_filename="final_readout.mp3"):
  #Combines a list of audio clips into a single MP3 file.
  #clip_paths: A list of file paths for the audio clips to be combined.
  #output_filename: The name for the final combined MP3 file.
    if not clip_paths:
        logging.warning("combine_audio_clips called with no clips to combine.")
        return None

    logging.info(f"--- Combining {len(clip_paths)} audio clips into '{output_filename}' ---")

    #Initialize an empty AudioSegment to build upon
    combined_audio = AudioSegment.empty()

    #Loop through each clip and append it to the combined audio
    for path in clip_paths:
        try:
            logging.info(f"   > Appending clip: {path}")
            # Load the audio clip (pydub can automatically detect the format)
            segment = AudioSegment.from_file(path)
            combined_audio += segment
        except pydub_exceptions.CouldntDecodeError:
            logging.error(f"Could not decode {path}. The file might be corrupted or empty. Skipping.")
            continue
        except FileNotFoundError:
            logging.error(f"File not found: {path}. Skipping.")
            continue

    if len(combined_audio) == 0:
        logging.error("Final combined audio is empty. Could not process any clips.")
        return None

    #Export the final combined audio to the specified format
    try:
        logging.info(f" > Exporting final audio to {output_filename}...")
        combined_audio.export(output_filename, format="mp3")
        logging.info(f"Successfully exported final audio file.")
    except Exception as e:
        logging.error(f"Failed to export combined audio file: {e}")
        logging.error("might be due to an issue with FFmpeg. Please check if it is installed and accessible in your system's path.")
        return None
      
    #After successfully exporting, delete the temporary individual clips
    logging.info("Cleaning up temporary audio clips...")
    for path in clip_paths:
        try:
            os.remove(path)
        except OSError as e:
            logging.warning(f"Could not remove temporary file {path}: {e}")
    logging.info("Cleanup complete")

    return output_filename

#This block is for testing the file directly
if __name__ == '__main__':
    #This code runs only when you execute python audio_combiner.py in your terminal
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("\n--- Running Test for audio_combiner.py ---")

    #For testing, we need to create some dummy audio files first.
    temp_test_dir = "temp_combiner_test"
    os.makedirs(temp_test_dir, exist_ok=True)
    
    print(f" > Creating dummy .wav files in '{temp_test_dir}'...")
    try:
        #Create 3 silent 1-second WAV files for the test
        dummy_clip_paths = []
        silent_segment = AudioSegment.silent(duration=1000) # 1000ms = 1 second
        for i in range(3):
            file_path = os.path.join(temp_test_dir, f"dummy_clip_{i+1}.wav")
            silent_segment.export(file_path, format="wav")
            dummy_clip_paths.append(file_path)
        
        print(f"Created {len(dummy_clip_paths)} dummy files.")

        #Now run the combination function on our dummy files
        test_output_file = "test_combined_output.mp3"
        result = combine_audio_clips(dummy_clip_paths, output_filename=test_output_file)

        if result and os.path.exists(test_output_file):
            print(f"SUCCESS: Audio combination test is passed!")
            print(f"Final file '{test_output_file}' was created.")
            
            #Check if cleanup worked
            clips_remain = any(os.path.exists(p) for p in dummy_clip_paths)
            if not clips_remain:
                print("Temporary dummy clips were successfully deleted.")
            else:
                print("Warning: Some temporary dummy clips were not deleted.")
            
            #Final cleanup
            os.remove(test_output_file)
            os.rmdir(temp_test_dir)
            print("Test output file and directory have been cleaned up.")
            print("-----------------------------------------------------")
        else:
            print("\n--- FAILURE: Audio combination test failed. Check logs for errors. ---")

    except Exception as e:
        print(f"\n--- An error occurred during the test: {e} ---")
        print("   This might mean pydub is not installed or FFmpeg is missing")
