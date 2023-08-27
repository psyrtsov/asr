from typing import List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from seamless_communication.models.inference import Translator
import tempfile



app = FastAPI()

class Segment(BaseModel):
    text: str
    no_speech_prob: float = 0.1

class Data(BaseModel):
    segments: List[Segment]

# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_large", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"), dtype=torch.float16)

@app.post("/asr")
def post_endpoint(audio_file: UploadFile=File(...)):
    with tempfile.NamedTemporaryFile("+bw") as f:
        f.write(audio_file.file.read())
        f.flush()
        transcribed_text = translator.predict(f.name, "asr", "eng")[0]
    return Data(segments=[Segment(text=str(transcribed_text))]) 

