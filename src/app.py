from fastapi import FastAPI
from fastapi.responses import FileResponse
import uuid
from generate_meme import generate_meme  # your existing function

app = FastAPI()

@app.get("/")
def home():
    return {"message": "MemeGAN API is running!"}

@app.get("/generate")
def generate(caption: str = "Funny meme"):
    output_path = f"outputs/meme_{uuid.uuid4().hex}.jpg"

    # call your existing meme generation pipeline
    generate_meme(caption, output_path)

    return FileResponse(output_path, media_type="image/jpeg")
