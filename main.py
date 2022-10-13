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

@app.get("/")
def root():
  return "Running" 

@app.post("/upload-file") 
async def process_exams(files: list[UploadFile], choicesCount: int = Form(5), correctAnswers: str = Form('[]'), id_digits_count: int = Form(10)):
  return process(list(map(to_bytes, files)), correct_answers = json.loads(correctAnswers), choices_per_question_count = choicesCount, id_digits_count = id_digits_count)

def to_bytes(file: UploadFile):
  return np.fromstring(file.file.read(), np.uint8)
