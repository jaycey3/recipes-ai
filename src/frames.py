import cv2
from PIL import Image
from typing import List, Tuple

def extract_frames(video_path: str, fps: float = 1.0) -> List[Tuple[float, Image.Image]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        raise ValueError(f"Could not read FPS from video file: {video_path}")
    
    interval = max(1, int(round(video_fps / fps)))

    frames = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            timestamp = frame_idx / video_fps
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append((timestamp, pil_image))

        frame_idx += 1

    cap.release()
    return frames

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python frames.py <video_path> [fps]")
        sys.exit(1)

    frames = extract_frames(sys.argv[1], fps=1.0)
    print(f"Extracted {len(frames)} frames")
    print(f"First frame at t={frames[0][0]:.2f}s, size={frames[0][1].size}")
    print(f"Last frame at t={frames[-1][0]:.2f}s")