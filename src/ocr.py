import easyocr
from PIL import Image
import numpy as np
from typing import List, Dict

LANG_MAP = {
    "nl": ["nl"],
    "en": ["en"],
    "auto": ["en", "nl"]
}

class OCRReader:
    def __init__(self, language: str = "en", use_gpu: bool = False):

        langs = LANG_MAP.get(language, ["en"])
        self.reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
    
    def read_frame(self, image: Image.Image, min_confidence: float = 0.4, min_length: int = 3) -> str:

        np_image = np.array(image)

        results = self.reader.readtext(np_image)

        snippets = []
        for _, text, confidence in results:
            text = text.strip()
            if confidence >= min_confidence and len(text) >= min_length:
                snippets.append(text)
        
        return " ".join(snippets)
    
def extract_text_from_frames(
        frames: List,
        language: str = "en",
        min_confidence: float = 0.4,
        min_length: int = 3,
        use_gpu: bool = False
) -> List[Dict]:
    
    reader = OCRReader(language=language, use_gpu=use_gpu)
    
    results = []
    for timestamp, image in frames:
        text = reader.read_frame(image, min_confidence=min_confidence, min_length=min_length)
        if text:
            results.append({
                "timestamp": timestamp,
                "text": text
            })
    return results