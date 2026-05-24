import subprocess
import os
import tempfile
import whisper
from typing import List, Dict

def extract_audio(video_path: str, audio_path: str = None) -> str:
    if audio_path is None:
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        audio_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")
    return audio_path

def transcribe(audio_path: str, model_size: str = "small", language: str = "en") -> List[Dict]:
    model = whisper.load_model(model_size)

    whisper_language = None if language == "auto" else language
    result = model.transcribe(audio_path, language=whisper_language)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip()
        })
    return segments

def transcribe_video(video_path: str, model_size: str = "small", language: str = "en") -> List[Dict]:
    audio_path = extract_audio(video_path)
    try:
        segments = transcribe(audio_path, model_size=model_size, language=language)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    return segments