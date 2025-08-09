import os
import time
import threading
import subprocess
import requests
from flask import Flask

app = Flask(__name__)

YOUTUBE_STREAM_KEY = os.getenv("YOUTUBE_STREAM_KEY")
FONT_PATH = "VT323-Regular.ttf"  # Keep this in repo root
PING_INTERVAL = 300  # seconds

# RTMP endpoint for YouTube Live ingestion
YOUTUBE_RTMP_URL = f"rtmp://a.rtmp.youtube.com/live2/{YOUTUBE_STREAM_KEY}"

def start_stream():
    """
    Start FFmpeg to push generated video to YouTube Live.
    This example sends a test pattern with text overlay.
    Replace the input section with your actual frame generation if needed.
    """
    ffmpeg_command = [
        "ffmpeg",
        "-re",
        "-f", "lavfi",
        "-i", "testsrc=size=1280x720:rate=30",
        "-vf", "drawtext=text='TerminalMind Live':fontcolor=white:fontsize=48:x=100:y=100",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-maxrate", "3000k",
        "-bufsize", "6000k",
        "-pix_fmt", "yuv420p",
        "-g", "60",
        "-f", "flv",
        YOUTUBE_RTMP_URL
    ]
    subprocess.Popen(ffmpeg_command)

def keep_alive():
    """
    Periodically ping the /ping endpoint so Render's free tier doesn't sleep.
    """
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
    return "TerminalMind stream is live to YouTube!"

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    # Start YouTube stream
    start_stream()

    # Start keep-alive thread
    threading.Thread(target=keep_alive, daemon=True).start()

    # Run Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
