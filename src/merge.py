from typing import List, Dict
from collections import Counter

def merge_detections_and_transcriptions(
        detections_per_frame: List[Dict], 
        transcript_segments: List[Dict], 
        ocr_per_frame: List[Dict] = None,
        min_detection_count: int = 1
) -> Dict:
    if ocr_per_frame is None:
        ocr_per_frame = []

    timeline = []
    for seg in transcript_segments:
        start, end = seg["start"], seg["end"]

        ingredients_in_segment = []
        for frame in detections_per_frame:
            if start <= frame["timestamp"] <= end:
                for det in frame["detections"]:
                    ingredients_in_segment.append(det["label_name"])

        counts = Counter(ingredients_in_segment)
        visible = [name for name, c in counts.items() if c >= min_detection_count]

        ocr_texts = []
        for ocr_frame in ocr_per_frame:
            if start <= ocr_frame["timestamp"] <= end:
                ocr_texts.append(ocr_frame["text"])

        timeline.append({
            "start": start,
            "end": end,
            "text": seg["text"],
            "visible_ingredients": visible,
            "on_screen_text": list(set(ocr_texts))
        })

    all_seen = set()
    for frame in detections_per_frame:
        for det in frame["detections"]:
            all_seen.add(det["label_name"])

    all_ocr_text = list(set(ocr["text"] for ocr in ocr_per_frame))

    duration = 0.0
    if detections_per_frame:
        duration = max(duration, detections_per_frame[-1]["timestamp"])
    if transcript_segments:
        duration = max(duration, transcript_segments[-1]["end"])

    return {
        "duration": duration,
        "all_ingredients": sorted(all_seen),
        "all_on_screen_text": all_ocr_text,
        "timeline": timeline
    }