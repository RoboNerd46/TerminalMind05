# app.py
"""
TerminalMind01 - optimized streaming app for Render (web service)
Streams a self-interviewing LLM to YouTube with CRT + typing animation.
"""

import os
import time
import threading
import subprocess
import requests
import random
import sys
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from flask import Flask, jsonify

app = Flask(__name__)

# -----------------------
# Config (env variables)
# -----------------------
MODEL = os.getenv("LLM7_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
API_URL = os.getenv("API_URL", "https://api.llm7.io/v1/chat/completions")

FONT_FILENAME = os.getenv("FONT_FILENAME", "VT323-Regular.ttf")
FONT_PATH = os.path.join(os.path.dirname(__file__), FONT_FILENAME)

YOUTUBE_STREAM_KEY = os.getenv("YOUTUBE_STREAM_KEY", "")
YOUTUBE_RTMP_URL = f"rtmp://a.rtmp.youtube.com/live2/{YOUTUBE_STREAM_KEY}"

# Visual/audio settings
WIDTH = int(os.getenv("WIDTH", "1280"))
HEIGHT = int(os.getenv("HEIGHT", "720"))
FPS = int(os.getenv("FPS", "30"))

# Display durations (seconds)
Q_DISPLAY = float(os.getenv("Q_SECONDS", "5.0"))   # after typing completes
A_DISPLAY = float(os.getenv("A_SECONDS", "6.0"))

# Typing effect params
TYPING_MIN_DELAY = float(os.getenv("TYPING_MIN_DELAY", "0.02"))  # per char
TYPING_MAX_DELAY = float(os.getenv("TYPING_MAX_DELAY", "0.09"))
PAUSE_AFTER_SENTENCE = float(os.getenv("PAUSE_AFTER_SENTENCE", "0.18"))

# Keep-alive (free tier)
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
PING_INTERVAL = int(os.getenv("PING_INTERVAL", "300"))  # seconds

# LLM constraints
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "60"))  # keep answers short

# Globals
STREAM_THREAD = None
STREAM_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
FFMPEG_PROCESS = None
IS_STREAMING = False

# -----------------------
# LLM helper (short Q/A)
# -----------------------
def query_llm7(prompt: str, model: str = MODEL, timeout: int = 30) -> str:
    """Call LLM7 and return assistant text. Defensive and concise."""
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": 0.9
        }
        resp = requests.post(API_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        # defensive navigation
        return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM] Error: {e}", file=sys.stderr)
        return f"(LLM error: {e})"

# Convenience prompts to ensure brevity and style
def ask_first_question() -> str:
    prompt = (
        "Please ask yourself a thought-provoking question that explores your own mind or idea-space. "
        "Keep the question concise (1-2 short sentences)."
    )
    return query_llm7(prompt)

def answer_question_short(q: str) -> str:
    prompt = (
        f"Answer this question in 1-2 short sentences (concise):\n\n{q}"
    )
    return query_llm7(prompt)

def next_question_from_answer(a: str) -> str:
    prompt = (
        f"Based on this short answer (one or two sentences):\n\n{a}\n\n"
        "Ask yourself the next deep or creative question to explore. Keep it to 1-2 short sentences."
    )
    return query_llm7(prompt)

# -----------------------
# Fast rendering helpers
# -----------------------
# Precompute scanline overlay as a numpy array to speed blend
def make_scanline_overlay(width:int, height:int, opacity:float=0.08, gap:int=2) -> np.ndarray:
    """Create an RGBA overlay with thin horizontal darker lines to simulate scanlines."""
    overlay = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA
    # Black horizontal lines at interval (gap + 1)
    for y in range(0, height, gap + 1):
        overlay[y:y+1, :, :3] = 0  # black line
        overlay[y:y+1, :, 3] = int(255 * opacity)
    return overlay

# Single font load
def load_font_sizes():
    """Return (font_heading, font_q, font_a) scaled to canvas size."""
    try:
        # heading large, q/a slightly smaller
        fh = ImageFont.truetype(FONT_PATH, size=max(28, int(HEIGHT * 0.08)))
        fq = ImageFont.truetype(FONT_PATH, size=max(20, int(HEIGHT * 0.06)))
        fa = ImageFont.truetype(FONT_PATH, size=max(18, int(HEIGHT * 0.05)))
        return fh, fq, fa
    except Exception as e:
        print(f"[font] Failed to load {FONT_PATH}: {e}. Falling back to default font.", file=sys.stderr)
        return ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default()

FONT_HEADING, FONT_Q, FONT_A = load_font_sizes()
SCANLINE_OVERLAY = make_scanline_overlay(WIDTH, HEIGHT, opacity=0.06, gap=2)

# Create base image once (heading + subtle vignette if desired)
def create_base_image() -> Image.Image:
    base = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(base)
    # Heading centered at top
    heading = "TerminalMind01"
    bbox = draw.textbbox((0,0), heading, font=FONT_HEADING)
    heading_w = bbox[2] - bbox[0]
    heading_x = (WIDTH - heading_w) // 2
    heading_y = int(HEIGHT * 0.03)
    draw.text((heading_x, heading_y), heading, font=FONT_HEADING, fill=(200, 255, 180))
    return base

BASE_IMAGE = create_base_image()

# Fast helper to draw text onto base and return BGR numpy for ffmpeg
def render_text_frame(question_text: str, answer_text: str, q_brightness:int=255, a_brightness:int=180) -> np.ndarray:
    """
    Draw question on one line and answer on the next.
    q_brightness and a_brightness control brightness (0-255).
    """
    # Copy base image (PIL)
    img = BASE_IMAGE.copy()
    draw = ImageDraw.Draw(img)

    margin_x = int(WIDTH * 0.06)
    # position below heading
    start_y = int(HEIGHT * 0.18)

    # wrap text to fit width
    def draw_wrapped(text, font, y, fill):
        # naive wrap: split into words and build lines that fit
        words = text.split()
        line = ""
        line_h = font.size if hasattr(font, "size") else int(HEIGHT * 0.05)
        for w in words:
            test = (line + " " + w).strip()
            bbox = draw.textbbox((0,0), test, font=font)
            if bbox[2] - bbox[0] > (WIDTH - 2 * margin_x) and line:
                draw.text((margin_x, y), line, font=font, fill=fill)
                y += int(line_h * 1.25)
                line = w
            else:
                line = test
        if line:
            draw.text((margin_x, y), line, font=font, fill=fill)
            y += int(line_h * 1.25)
        return y

    # Brightness -> color
    q_color = (0, q_brightness, 0)
    a_color = (0, a_brightness, 0)

    y_after_q = draw_wrapped(question_text, FONT_Q, start_y, q_color)
    _ = draw_wrapped(answer_text, FONT_A, y_after_q + int(FONT_Q.size * 0.3), a_color)

    # convert PIL to numpy BGR quickly
    arr = np.array(img, dtype=np.uint8)

    # Apply scanline overlay (RGBA) onto arr
    # overlay alpha blending: out = (1-alpha)*arr + alpha*overlay_rgb
    alpha = (SCANLINE_OVERLAY[...,3:4].astype(np.float32) / 255.0)
    overlay_rgb = SCANLINE_OVERLAY[..., :3].astype(np.float32)
    arr = arr.astype(np.float32)
    arr = (arr * (1.0 - alpha) + overlay_rgb * alpha).astype(np.uint8)

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

# -----------------------
# FFmpeg process helper
# -----------------------
def start_ffmpeg():
    global FFMPEG_PROCESS
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-b:v", "3000k",
        "-maxrate", "3000k",
        "-bufsize", "6000k",
        "-g", str(FPS * 2),
        "-c:a", "aac",
        "-ar", "44100",
        "-ac", "2",
        "-f", "flv",
        YOUTUBE_RTMP_URL
    ]
    print(f"[ffmpeg] starting ffmpeg...", file=sys.stderr)
    FFMPEG_PROCESS = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    return FFMPEG_PROCESS

def stop_ffmpeg():
    global FFMPEG_PROCESS
    try:
        if FFMPEG_PROCESS:
            print("[ffmpeg] terminating...", file=sys.stderr)
            FFMPEG_PROCESS.terminate()
            try:
                FFMPEG_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                FFMPEG_PROCESS.kill()
    except Exception as e:
        print(f"[ffmpeg] stop error: {e}", file=sys.stderr)
    finally:
        FFMPEG_PROCESS = None

# -----------------------
# Streaming loop (self-interview + typing)
# -----------------------
def typing_and_stream(proc, text_full, is_question=True):
    """
    Simulate typing for text_full. Write frames to proc.stdin for each new char.
    Returns when typing complete.
    """
    # brightness: question brighter than answer
    q_brightness = 240
    a_brightness = 170

    shown = ""
    for i, ch in enumerate(text_full):
        if STOP_EVENT.is_set():
            return
        shown += ch
        # For speed: update only when a new printable char added (we are already at char-level)
        q_text = shown if is_question else ""
        a_text = "" if is_question else shown

        # When typing question, show blank answer (or last full answer) — for simplicity we leave answer blank while question types
        if is_question:
            frame = render_text_frame(shown, "")  # q on first line, a empty
        else:
            # if answer typing, show full question (we pass empty; outer logic keeps last q)
            frame = render_text_frame(current_q_global, shown, q_brightness, a_brightness)

        try:
            proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            STOP_EVENT.set()
            return

        # Natural typing pause
        delay = random.uniform(TYPING_MIN_DELAY, TYPING_MAX_DELAY)
        # Slightly longer pause after punctuation/sentence ends
        if ch in ".!?":
            delay += PAUSE_AFTER_SENTENCE
        time.sleep(delay)

    # finished typing; short settle
    time.sleep(0.05)


# We'll hold the current question globally for answer-type frames
current_q_global = ""

def streaming_loop():
    global IS_STREAMING, current_q_global
    print("[stream] starting streaming loop...", file=sys.stderr)
    proc = None
    try:
        proc = start_ffmpeg()
    except Exception as e:
        print(f"[stream] cannot start ffmpeg: {e}", file=sys.stderr)
        IS_STREAMING = False
        return

    # initial Q
    q = ask_first_question()
    current_q_global = q

    while not STOP_EVENT.is_set():
        # short answer loop
        try:
            # Ensure q is short — we already asked LLM to be concise, but truncate defensively
            q = q.strip()
            if len(q) > 300:
                q = q[:300].rsplit('.', 1)[0] + "."

            current_q_global = q

            # TYPE QUESTION
            typing_and_stream(proc, q, is_question=True)

            # After typing question, hold Q on-screen for the remainder of Q_DISPLAY (we already spent typing time)
            # Compute how many frames to show (approx)
            q_frames = max(1, int(FPS * Q_DISPLAY))
            frame_q = render_text_frame(q, "")
            for _ in range(q_frames):
                if STOP_EVENT.is_set(): break
                try:
                    proc.stdin.write(frame_q.tobytes())
                except BrokenPipeError:
                    STOP_EVENT.set()
                    break
                time.sleep(1.0 / FPS)

            if STOP_EVENT.is_set(): break

            # GET SHORT ANSWER
            a = answer_question_short(q)
            if len(a) > 400:
                a = a[:400].rsplit('.', 1)[0] + "."

            # TYPE ANSWER
            typing_and_stream(proc, a, is_question=False)

            # After typing answer, hold A on-screen for A_DISPLAY
            a_frames = max(1, int(FPS * A_DISPLAY))
            frame_qa = render_text_frame(q, a)
            for _ in range(a_frames):
                if STOP_EVENT.is_set(): break
                try:
                    proc.stdin.write(frame_qa.tobytes())
                except BrokenPipeError:
                    STOP_EVENT.set()
                    break
                time.sleep(1.0 / FPS)

            # produce next question from answer
            q = next_question_from_answer(a)
            # small safeguard
            if not q or "error" in q.lower():
                q = ask_first_question()

            current_q_global = q

        except Exception as e:
            print(f"[stream] loop error: {e}", file=sys.stderr)
            # wait briefly and continue to avoid rapid crash loops
            time.sleep(1)
            continue

    # cleanup
    stop_ffmpeg()
    IS_STREAMING = False
    STOP_EVENT.clear()
    print("[stream] ended.", file=sys.stderr)

# -----------------------
# Keep-alive pinger
# -----------------------
def keep_alive_loop():
    if not RENDER_EXTERNAL_URL:
        print("[keep-alive] RENDER_EXTERNAL_URL not set; disabling keep-alive.", file=sys.stderr)
        return
    ping_url = RENDER_EXTERNAL_URL.rstrip("/") + "/ping"
    print(f"[keep-alive] will ping {ping_url} every {PING_INTERVAL}s", file=sys.stderr)
    while True:
        try:
            requests.get(ping_url, timeout=10)
            print("[keep-alive] pinged", ping_url, file=sys.stderr)
        except Exception as e:
            print("[keep-alive] ping failed:", e, file=sys.stderr)
        time.sleep(PING_INTERVAL)

# -----------------------
# Flask endpoints (control)
# -----------------------
@app.route("/")
def index():
    return jsonify({
        "service": "TerminalMind01 self-interview stream",
        "is_streaming": IS_STREAMING,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "fps": FPS
    })

@app.route("/ping")
def ping():
    return "pong"

@app.route("/status")
def status():
    return jsonify({
        "is_streaming": IS_STREAMING,
        "ffmpeg_running": FFMPEG_PROCESS is not None and (FFMPEG_PROCESS.poll() is None),
        "youtube_configured": bool(YOUTUBE_STREAM_KEY)
    })

@app.route("/stream")
def start_stream_route():
    global STREAM_THREAD, IS_STREAMING, STOP_EVENT
    if not YOUTUBE_STREAM_KEY:
        return "Missing YOUTUBE_STREAM_KEY env var", 500
    with STREAM_LOCK:
        if IS_STREAMING:
            return "already_streaming", 200
        STOP_EVENT.clear()
        STREAM_THREAD = threading.Thread(target=streaming_loop, daemon=True)
        STREAM_THREAD.start()
        IS_STREAMING = True
        print("[/stream] started", file=sys.stderr)
        return "stream_started", 200

@app.route("/stop")
def stop_stream_route():
    global STREAM_THREAD, IS_STREAMING
    with STREAM_LOCK:
        if not IS_STREAMING:
            return "not_streaming", 200
        print("[/stop] stop requested", file=sys.stderr)
        STOP_EVENT.set()
        # wait a short bit
        start = time.time()
        timeout = 12
        while STREAM_THREAD and STREAM_THREAD.is_alive() and (time.time() - start) < timeout:
            time.sleep(0.2)
        IS_STREAMING = False
        return "stop_requested", 200

# -----------------------
# Main: start keep-alive and Flask
# -----------------------
if __name__ == "__main__":
    # start keep-alive (daemon)
    if RENDER_EXTERNAL_URL:
        threading.Thread(target=keep_alive_loop, daemon=True).start()

    # do not auto-start streaming on boot; start with /stream
    port = int(os.environ.get("PORT", "5000"))
    print(f"[main] Flask starting on 0.0.0.0:{port}", file=sys.stderr)
    app.run(host="0.0.0.0", port=port)
