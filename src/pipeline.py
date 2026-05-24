import os
import json
from typing import Dict, Optional, Callable

from src.frames import extract_frames
from src.detection import IngredientDetector
from src.audio import transcribe_video
from src.merge import merge_detections_and_transcriptions
from src.llm import structure_recipe
from src.ocr import extract_text_from_frames

DEFAULT_MODEL_PATH = "models/faster_rcnn_model.pth"
DEFAULT_FRAME_FPS = 1.0
DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_WHISPER_SIZE = "small"
DEFAULT_LANGUAGE = "en"

CATEGORY_NAMES = {
    0: "food",
    1: "apple",
    # 2: "flour",
    # 3: "sugar",
    # 4: "butter",
    # 5: "egg",
    # 6: "cinnamon"
}

NUM_CLASSES = len(CATEGORY_NAMES)

def process_video(
        video_path: str, 
        model_path: str = DEFAULT_MODEL_PATH,
        frame_fps: float = DEFAULT_FRAME_FPS,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
        whisper_size: str = DEFAULT_WHISPER_SIZE,
        language: str = DEFAULT_LANGUAGE,
        progress_callback: Optional[Callable[[str], None]] = None
        ) -> Dict:
    
    def progress(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    progress("Extracting frames from video...")
    frames = extract_frames(video_path, fps=frame_fps)
    progress(f"Extracted {len(frames)} frames.")

    progress("Loading detection model...")
    detector = IngredientDetector(model_path=model_path, num_classes=NUM_CLASSES, category_names=CATEGORY_NAMES)
    progress(f"Running detection on {len(frames)} frames...")
    detections_per_frame = []
    for timestamp, image in frames:
        detections = detector.predict(image, threshold=detection_threshold)
        detections_per_frame.append({
            "timestamp": timestamp,
            "detections": detections
        })

    progress("Extracting on-screen text with OCR...")
    ocr_per_frame = extract_text_from_frames(frames, language=language)
    progress(f"Extracted on-screen text from {len(ocr_per_frame)} frames.")

    progress("Transcribing audio...")
    transcript_segments = transcribe_video(video_path, model_size=whisper_size, language=language)
    progress(f"Transcribed {len(transcript_segments)} segments.")

    progress("Merging detections, transcriptions, and OCR text...")
    merged = merge_detections_and_transcriptions(detections_per_frame, transcript_segments, ocr_per_frame=ocr_per_frame)

    progress("Structuring recipe...")
    recipe = structure_recipe(merged, language=language)

    recipe["_debug"] = {
        "num_frames": len(frames),
        "num_transcript_segments": len(transcript_segments),
        "merged_input": merged
    }

    progress("Done.")
    return recipe

def save_recipe(recipe: Dict, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print ("Usage: python -m src.pipeline <video_path>")
        sys.exit(1)

    video = sys.argv[1]
    recipe = process_video(video)

    out_path = os.path.join("data/output_recipes", os.path.splitext(os.path.basename(video))[0] + ".json")
    save_recipe(recipe, out_path)
    print(f"\nRecipe saved to {out_path}")