# from fastapi import FastAPI
# from fastapi.responses import FileResponse
# import uuid
# from generate_meme import generate_meme  # your existing function
#
# app = FastAPI()
#
# @app.get("/")
# def home():
#     return {"message": "MemeGAN API is running!"}
#
# @app.get("/generate")
# def generate(caption: str = "Funny meme"):
#     output_path = f"outputs/meme_{uuid.uuid4().hex}.jpg"
#
#     # call your existing meme generation pipeline
#     generate_meme(caption, output_path)
#
#     return FileResponse(output_path, media_type="image/jpeg")

# src/app.py
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

# make sure outputs folder exists
os.makedirs("outputs", exist_ok=True)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "MemeGAN API is running!"}

@app.get("/generate")
def generate(caption: str = "Funny meme"):
    # lazy import to avoid import-time failures when heavy ML libs are missing
    try:
        from .generate_meme import generate_meme
    except Exception as e:
        # return a clear HTTP error instead of crashing the app on import
        raise HTTPException(status_code=500, detail=f"Could not import generate_meme: {e!s}")

    output_path = f"outputs/meme_{uuid.uuid4().hex}.jpg"

    try:
        generate_meme(caption, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e!s}")

    return FileResponse(output_path, media_type="image/jpeg")


if __name__ == "__main__":
    # local-run convenience
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

