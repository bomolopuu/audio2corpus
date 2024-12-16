import streamlit as st
import tempfile
import os
from pathlib import Path
import torch
from transformers import pipeline

#CHANGE THE MODEL, HERE IS AN EXAMPLE
def load_transcriber():
    transcriber = pipeline("automatic-speech-recognition", model = "mms-meta/mms-zeroshot-300m" )
    return transcriber

def transcribe_audio(audio_file, transcriber):
    try:
        # Transcribe the audio
        result = transcriber(audio_file)
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def main():
    st.title("Audio Transcription App")
    st.write("Upload an audio file to get its transcription")

    # File uploader
    audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a'])

    if audio_file is not None:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name

        # Display audio player
        st.audio(audio_file)

        # Add transcribe button
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing..."):
                # Load the transcriber
                transcriber = load_transcriber()

                # Get transcription
                transcription = transcribe_audio(audio_path, transcriber)

                if transcription:
                    st.success("Transcription Complete!")
                    st.write("Transcription:")
                    st.write(transcription)

        # Clean up temporary file
        try:
            os.unlink(audio_path)
        except:
            pass

if __name__ == "__main__":
    main()
