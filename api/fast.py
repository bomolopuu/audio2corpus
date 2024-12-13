from fastapi import FastAPI, File, UploadFile
from transformers import AutoProcessor, AutoModelForCTC
import librosa
import os
import torch
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#load model
processor = AutoProcessor.from_pretrained("mms-meta/mms-zeroshot-300m")
model = AutoModelForCTC.from_pretrained("mms-meta/mms-zeroshot-300m")
app.state.model = model

UPLOAD_DIR="uploaded_audio"
os.makedirs(UPLOAD_DIR,exist_ok=True)

@app.post("/transcribe/")
async def transcribe(audio_file: UploadFile = File(...)
                    #  vocab_file: UploadFile = File(...)
                     ):

    # Save the uploaded audio file
    audio_path = os.path.join(UPLOAD_DIR,audio_file.filename)
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())

    # # Save the uploaded vocabulary file
    # vocab_path = f"temp_{vocab_file.filename}"
    # with open(vocab_path, "wb") as f:
    #     f.write(await vocab_file.read())

    # Load the audio
    waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    #preprocess the audio
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    model=app.state.model

    # Device Management
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = inputs.to(device)

    # Perform transcription
    with torch.no_grad():
            logits = model(inputs.input_values).logits

    # Decode the logits to transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Clean up the temporary file
    # os.remove(audio_path)
    # os.remove(vocab_path)

    return {"transcription": transcription, "filename": audio_file.filename}
    # return FileResponse(audio_path)

    # return {"audio": {audio_file.},
    #         "vocabulary": {vocab_file.filename}}

@app.get("/media/{file_name}")
async def return_media(file_name):
#     return FileResponse(os.path.join("uploaded_audio", file_name))

# Find the audio file in the storage directory
    for files in os.listdir(UPLOAD_DIR):
        if files.startswith(file_name):
            file_path = os.path.join(UPLOAD_DIR, files)
            return FileResponse(file_path, media_type="audio/mpeg", filename=files)

    return {"error": "Audio file not found"}


@app.get("/")
def root():
    return {'message': 'API is working'}
