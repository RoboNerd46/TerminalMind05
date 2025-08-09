import os
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
import threading
from flask import Flask
import time

app = Flask(__name__)

# Config
MODEL = os.getenv("LLM7_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
API_URL = "https://api.llm7.io/v1/chat/completions"
FONT_PATH = "static/VT323-Regular.ttf"
YOUTUBE_STREAM_KEY = os.getenv("YOUTUBE_STREAM_KEY")
STREAM_URL = f"rtmp://a.rtmp.youtube.com/live2/{YOUTUBE_STREAM_KEY}"

# Function to query LLM7
def query_llm7(prompt, model=MODEL):
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 150}
    response = requests.post(API_URL, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Create frame with text
def render_frame(text, width=1920, height=1080):
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 48)
    draw.text((50, 50), text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Streaming loop
def stream_to_youtube():
    # FFmpeg process
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "1920x1080",
        "-r", "30",
        "-i", "-",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-b:v", "4000k",
        "-c:a", "aac",
        "-ar", "44100",
        "-f", "flv",
        STREAM_URL
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    while True:
        try:
            q = "What is consciousness?"
            a = query_llm7(q)
            frames = [
                render_frame(f"Q: {q}"),
                render_frame(f"A: {a}")
            ]
            for frame in frames:
                process.stdin.write(frame.tobytes())
                time.sleep(2)  # 2 seconds per frame
        except Exception as e:
            print(f"Error in streaming loop: {e}")
            break

# Start streaming in background
def start_streaming():
    thread = threading.Thread(target=stream_to_youtube, daemon=True)
    thread.start()

@app.route("/")
def index():
    return "TerminalMind streaming service is running."

if __name__ == "__main__":
    start_streaming()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
