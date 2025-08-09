import os
import time
import threading
import subprocess
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask

# Flask app
app = Flask(__name__)

# Environment variables
YOUTUBE_STREAM_KEY = os.getenv("YOUTUBE_STREAM_KEY")
FONT_PATH = "VT323-Regular.ttf"  # Keep this in repo root
PING_INTERVAL = 300  # seconds
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
API_URL = "https://api.llm7.io/v1/chat/completions"

# RTMP endpoint for YouTube Live ingestion
YOUTUBE_RTMP_URL = f"rtmp://a.rtmp.youtube.com/live2/{YOUTUBE_STREAM_KEY}"

def query_llm7(prompt):
    """Query LLM7 API (no API key needed)."""
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
    }
    response = requests.post(API_URL, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def render_frame(text, width=1280, height=720):
    """Render a single text frame with CRT green-on-black style."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 36)
    draw.text((50, 50), text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def start_stream():
    """Stream generated Q&A frames directly to YouTube Live."""
    q = "What is consciousness?"
    a = query_llm7(q)

    # Create a video stream via FFmpeg using pipe
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "1280x720",
        "-r", "30",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-maxrate", "3000k",
        "-bufsize", "6000k",
        "-pix_fmt", "yuv420p",
        "-g", "60",
        "-f", "flv",
        YOUTUBE_RTMP_URL
    ]

    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    # Loop frames to keep stream alive
    while True:
        frames = [
            render_frame(f"Q: {q}"),
            render_frame(f"A: {a}")
        ]
        for frame in frames:
            process.stdin.write(frame.tobytes())
        time.sleep(2)  # Delay between Q&A pairs

def keep_alive():
    """Periodically ping the /ping endpoint so Render's free tier stays awake."""
    url = os.getenv("RENDER_EXTERNAL_URL")
    if not url:
        print("No RENDER_EXTERNAL_URL set — keep-alive disabled.")
        return

    while True:
        try:
            print(f"Pinging {url}/ping")
            requests.get(f"{url}/ping", timeout=10)
        except Exception as e:
            print(f"Keep-alive ping failed: {e}")
        time.sleep(PING_INTERVAL)

@app.route("/")
def index():
    return "TerminalMind YouTube Live stream is running."

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    # Start streaming in background thread
    threading.Thread(target=start_stream, daemon=True).start()

    # Start keep-alive thread
    threading.Thread(target=keep_alive, daemon=True).start()

    # Run Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
