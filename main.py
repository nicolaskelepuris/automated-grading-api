from itertools import count
from urllib import request
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import controller
import os 


dir_path = os.path.dirname(os.path.realpath(__file__))

isdir = os.path.isdir(f'{dir_path}\\images')

if not isdir:
  IMAGEDIR = os.mkdir(f'{dir_path}\\images')

fullpath = os.path.join(dir_path, "images")

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
    for file in files:
      contents = await file.read()
      with open(f"{fullpath}\\{file.filename}", "wb") as f:
        f.write(contents)
      filesList.append(contents)
      finalImg = controller.read(f"{fullpath}\\{file.filename}")
      controller.show(f'{file.filename}', finalImg)
      controller.wait(0)
    return {"uploadStatus" : "Complete"}



      #img = Image.open(io.BytesIO(im))
      #print([f"Name: {file.filename}" for file in files])
