from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from controller import process
import numpy as np
import json

app = FastAPI()
filesList = []

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.post("/upload-file") 
async def create_upload_file(files: list[UploadFile], choicesCount: int = Form(5), correctAnswers: str = Form('[]')):
  return process(list(map(to_bytes, files)), correct_answers = json.loads(correctAnswers), choices_per_question_count = choicesCount)

def to_bytes(file: UploadFile):
  return np.fromstring(file.file.read(), np.uint8)
