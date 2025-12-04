import sys
from inference import run_inference

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
    else:
        sentence = "When your code finally runs without errors"

    output = run_inference(sentence)
    print(f"✅ Meme generated and saved at: {output}")
