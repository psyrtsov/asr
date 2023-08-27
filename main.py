from typing import List
from urllib import request
from fastapi import FastAPI, File, Response, UploadFile
from pydantic import BaseModel
import torch
from seamless_communication.models.inference import Translator
import tempfile

import torchaudio



app = FastAPI()

class Segment(BaseModel):
    text: str
    no_speech_prob: float = 0.1

class Data(BaseModel):
    segments: List[Segment]

# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_large", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"), dtype=torch.float16)

@app.post("/asr")
def asr(lang: str = 'eng', audio_file: UploadFile=File(...)):
    with tempfile.NamedTemporaryFile("+bw") as f:
        f.write(audio_file.file.read())
        f.flush()
        transcribed_text = translator.predict(f.name, "asr", "eng")[0]
    return Data(segments=[Segment(text=str(transcribed_text))]) 

class RequestBody(BaseModel):
    text: str
    lang: str = 'eng'
    
@app.post("/t2st")
def t2st(body: RequestBody):
    wav = translator.predict(body.text, "t2st", body.lang, src_lang=body.lang)
    with tempfile.NamedTemporaryFile("+bw", suffix='.wav') as f:
        torchaudio.save(
            f.name,
            wav[1][0].cpu(),
            sample_rate=wav[2]
        )
        f.flush()
        content = open(f.name, 'rb').read()
    return Response(content=content, media_type="audio/wav")
