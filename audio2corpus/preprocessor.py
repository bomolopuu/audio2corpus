from pydub import AudioSegment
import os
from pathlib import Path
import librosa

"""
Audio preprocessing module for transcription API.
Handles format conversion, sample rate adjustment, and audio splitting.
"""

def split_audio(audio, segment_length=30000):
    """
    Splits an audio segment into multiple smaller segments of equal length.

    Parameters:
        audio (AudioSegment): The audio to be split
        segment_length (int): Length of each segment in milliseconds

    Returns:
        list: List of paths to temporary WAV files containing the segments
    """
    segments = []
    audio_length = len(audio)
    temp_paths = []

    for start in range(0, audio_length, segment_length):
        end = min(start + segment_length, audio_length)
        segment = audio[start:end]

        # Save segment to temporary file
        temp_path = f"temp_segment_{len(temp_paths):02d}.wav"
        segment.export(temp_path, format='wav')
        temp_paths.append(temp_path)

    return temp_paths

def preprocess_audio(input_path, target_sr=16000, max_duration=30):
    """
    Preprocesses audio file for transcription.

    Parameters:
        input_path (str): Path to input audio file (can be temporary)
        target_sr (int): Target sample rate in Hz
        max_duration (int): Maximum duration of each segment in seconds

    Returns:
        tuple: (list of waveforms, list of temp files to clean up)
    """
    try:
        print(f"Processing file: {input_path}")
        temp_files = []
        waveforms = []

        # Load the audio file using pydub
        audio = AudioSegment.from_file(input_path)

        # Convert sample rate if needed
        if audio.frame_rate != target_sr:
            print(f"Converting to {target_sr} Hz...")
            audio = audio.set_frame_rate(target_sr)

        duration = len(audio) / 1000
        print(f"Audio duration: {duration:.2f} seconds")

        if duration > max_duration:
            print(f"Audio longer than {max_duration} seconds, splitting into parts...")
            segment_paths = split_audio(audio, max_duration * 1000)
            temp_files.extend(segment_paths)

            # Load each segment with librosa
            for path in segment_paths:
                waveform, _ = librosa.load(path, sr=target_sr, mono=True)
                waveforms.append(waveform)
        else:
            # For short audio, process directly
            temp_path = "temp_processed.wav"
            audio.export(temp_path, format='wav')
            temp_files.append(temp_path)

            waveform, _ = librosa.load(temp_path, sr=target_sr, mono=True)
            waveforms.append(waveform)

        return waveforms, temp_files

    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        # Clean up any temporary files that might have been created
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        raise

def cleanup_temp_files(temp_files):
    """
    Removes temporary files created during preprocessing.

    Parameters:
        temp_files (list): List of temporary file paths to remove
    """
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up temporary file: {temp_file}")
