from pydub import AudioSegment
import os
from pathlib import Path

"""
This program performs three main operations on audio files:
1. Converts any audio format to WAV
2. Changes the sample rate to 16kHz
3. Splits audio files longer than 30 seconds (or specified duration) into smaller segments

The program uses the pydub library, which internally uses ffmpeg for audio processing.
Supported input formats include: MP3, WAV, OGG, FLAC, M4A, WMA, AAC
"""

def split_audio(audio, segment_length=30000):
    """
    Splits an audio segment into multiple smaller segments of equal length.

    Parameters:
        audio (AudioSegment): The audio to be split
        segment_length (int): Length of each segment in milliseconds
                            Default is 30000ms (30 seconds)

    Returns:
        list: A list containing multiple AudioSegment objects, each representing
              a portion of the original audio

    Note: The last segment might be shorter than segment_length if the audio
          length is not perfectly divisible by segment_length
    """
    segments = []
    audio_length = len(audio)  # Total length in milliseconds

    # Iterate through the audio, taking segment_length chunks
    # The range function will automatically handle the last shorter segment
    for start in range(0, audio_length, segment_length):
        # Calculate end point, ensuring we don't exceed audio length
        end = min(start + segment_length, audio_length)
        # Extract segment using array slicing
        segment = audio[start:end]
        segments.append(segment)

    return segments

def convert_audio(input_path, output_path=None, target_sr=16000, max_duration=30):
    """
    Main conversion function that processes a single audio file.

    Workflow:
    1. Loads the audio file (any supported format)
    2. Converts sample rate to target rate (default 16kHz)
    3. If audio is longer than max_duration, splits it into parts
    4. Saves result(s) as WAV file(s)

    Parameters:
        input_path (str): Full path to the input audio file
        output_path (str): Where to save the processed file (optional)
                          If not provided, creates name based on input file
        target_sr (int): Target sample rate in Hz (default 16000)
        max_duration (int): Maximum duration of each segment in seconds (default 30)

    Returns:
        bool: True if conversion was successful, False if any error occurred
    """
    try:
        print(f"Processing file: {input_path}")

        # Load the audio file using pydub
        # AudioSegment.from_file automatically detects the format based on file extension
        audio = AudioSegment.from_file(input_path)

        # Get the current sample rate of the loaded audio
        current_sr = audio.frame_rate
        print(f"Current sample rate: {current_sr} Hz")

        # If the current sample rate doesn't match our target,
        # resample the audio to the target rate
        if current_sr != target_sr:
            print(f"Converting to {target_sr} Hz...")
            audio = audio.set_frame_rate(target_sr)

        # Calculate duration in seconds
        # AudioSegment stores length in milliseconds, so divide by 1000
        duration = len(audio) / 1000
        print(f"Audio duration: {duration:.2f} seconds")

        # If audio is longer than max_duration, we need to split it
        if duration > max_duration:
            print(f"Audio longer than {max_duration} seconds, splitting into parts...")
            # Get list of segments
            segments = split_audio(audio, max_duration * 1000)

            # Create the base path for our output files
            # If no output path provided, use input path without extension
            if output_path is None:
                base_path = Path(input_path).with_suffix('')
            else:
                base_path = Path(output_path).with_suffix('')

            # Save each segment as a separate WAV file
            # Enumerate from 1 to get part numbers starting at 1
            for i, segment in enumerate(segments, 1):
                # Create filename with part number (e.g., _part01, _part02)
                segment_path = f"{base_path}_part{i:02d}.wav"
                segment.export(segment_path, format='wav')
                print(f"Saved segment {i}: {segment_path}")
        else:
            # For short audio files, just save as a single file
            if output_path is None:
                # If no output path provided, add _16khz to original filename
                output_path = str(Path(input_path).with_suffix('_16khz.wav'))
            audio.export(output_path, format='wav')
            print(f"File saved: {output_path}")

        return True

    except Exception as e:
        # If anything goes wrong during processing, print error and return False
        print(f"Error converting file: {e}")
        return False

def batch_convert_directory(input_dir, output_dir=None, max_duration=30):
    """
    Processes all supported audio files in a directory.

    This function:
    1. Scans input directory for audio files
    2. Creates output directory if it doesn't exist
    3. Processes each supported audio file
    4. Maintains directory structure in output

    Parameters:
        input_dir (str): Directory containing audio files to process
        output_dir (str): Directory where processed files will be saved
                         If None, creates 'converted' subdirectory in input_dir
        max_duration (int): Maximum duration for audio segments in seconds
    """
    # If no output directory specified, create 'converted' subdirectory
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'converted')

    # Create output directory if it doesn't exist
    # exist_ok=True prevents errors if directory already exists
    os.makedirs(output_dir, exist_ok=True)

    # List of file extensions that our program can handle
    # These are the most common audio formats supported by ffmpeg
    supported_formats = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.wma', '.aac']

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        # Check if file has a supported extension
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            # Create full path for input and output files
            input_path = os.path.join(input_dir, filename)
            output_base = os.path.join(output_dir, Path(filename).stem)
            # Process the file
            convert_audio(input_path, output_base, max_duration=max_duration)

# Example usage section
if __name__ == "__main__":
    """
    This section demonstrates how to use the program.
    It only runs when the script is executed directly (not when imported as a module).
    """

    # Example 1: Convert a single file
    audio_file = "path/to/your/audio.mp3"
    convert_audio(audio_file, max_duration=30)

    # Example 2: Convert all files in a directory
    directory = "path/to/your/audio/folder"
    batch_convert_directory(directory, max_duration=30)
