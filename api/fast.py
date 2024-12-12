from fastapi import FastAPI, File, UploadFile
from transformers import AutoProcessor, AutoModelForCTC
import librosa
import os
import torch

app = FastAPI()

#load model
# processor = AutoProcessor.from_pretrained("mms-meta/mms-zeroshot-300m")
# model = AutoModelForCTC.from_pretrained("mms-meta/mms-zeroshot-300m")
# app.state.model = model

@app.post("/transcribe/")
async def transcribe(audio_file: UploadFile = File(...),
                     vocab_file: UploadFile = File(...)
                     ):

    # Save the uploaded audio file
    audio_path = f"temp_{audio_file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())

    # Save the uploaded vocabulary file
    vocab_path = f"temp_{vocab_file.filename}"
    with open(vocab_path, "wb") as f:
        f.write(await vocab_file.read())

    # # Load the audio
    # waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    # #preprocess the audio
    # inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)


    # # Device Management
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    # inputs = inputs.to(device)

    # # Perform transcription
    # with torch.no_grad():
    #         logits = model(inputs.input_values).logits

    # # Decode the logits to transcription
    # predicted_ids = torch.argmax(logits, dim=-1)
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Clean up the temporary file
    os.remove(audio_path)
    os.remove(vocab_path)

    # return {"transcription": transcription}

    return {"audio": {audio_file.filename},
            "vocabulary": {vocab_file.filename}}


@app.get("/")
def root():
    return {'message': 'API is working'}
