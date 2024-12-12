from fastapi import FastAPI, File, UploadFile
from transformers import AutoProcessor, AutoModelForCTC
import librosa
import os
import torch

app = FastAPI()

#load model
processor = AutoProcessor.from_pretrained("mms-meta/mms-zeroshot-300m")
model = AutoModelForCTC.from_pretrained("mms-meta/mms-zeroshot-300m")
# app.state.model = model

@app.post("/transcribe/")
async def transcribe(audio_file: UploadFile = File(...),
                     csv_file: UploadFile = File(...)
                     ):

    # Save the uploaded file locally
    file_location = f"temp_{audio_file.filename}"
    with open(file_location, "wb") as f:
        f.write(await audio_file.read())

    # Save the uploaded csv_file locally
    csv_location = f"temp_{csv_file.filename}"
    with open(csv_location, "wb") as f:
        f.write(await csv_file.read())

    # # Load audio using librosa
    # waveform, sample_rate = librosa.load(file_location, sr=16000, mono=True)

    # # Convert waveform to tensor
    # waveform_tensor = torch.tensor(waveform).unsqueeze(0)

    # Preprocess the audio
    # inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Generate transcription logits
    # with torch.no_grad():
    #     logits = model(inputs.input_values).logits

    # Decode the logits to text
    # predicted_ids = torch.argmax(logits, dim=-1)
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Clean up the temporary file
    os.remove(file_location)
    os.remove(csv_location)

    # return {"transcription": transcription}

    return {"audio": {audio_file.filename},
            "vocabulary": {csv_file.filename}}


@app.get("/")
def root():
    return {'message': 'API is working'}
