import os
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", token=HF_TOKEN
)
pipeline.to(torch.device("mps"))
output = pipeline("audio.wav")
print(dir(output))
