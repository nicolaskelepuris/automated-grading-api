from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from controller import proccess
import numpy as np

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
async def create_upload_file(files: list[UploadFile]):
    proccess(list(map(to_bytes, files)))
    return { "uploadStatus" : "Complete" }

def to_bytes(file: UploadFile):
    return np.fromstring(file.file.read(), np.uint8)
