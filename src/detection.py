import os
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from typing import List, Dict

class IngredientDetector:
    def __init__(self, model_path: str, num_classes: int, category_names: Dict[int, str], device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.category_names = category_names

        self.model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Train the model first using the notebook."
            )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.ToTensor()

    def predict(self, image: Image.Image, threshold: float = 0.5) -> List[Dict]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)[0]

        keep = outputs["scores"] > threshold
        boxes = outputs["boxes"][keep].cpu().tolist()
        labels = outputs["labels"][keep].cpu().tolist()
        scores = outputs["scores"][keep].cpu().tolist()

        detections = []
        for box, label, score in zip(boxes, labels, scores):
            detections.append({
                "label_id": label,
                "label_name": self.category_names.get(label, str(label)),
                "score": float(score),
                "box": box
            })
        return detections
    
    def predict_batch(self, images: List[Image.Image], threshold: float = 0.5) -> List[List[Dict]]:
        return [self.predict(image, threshold=threshold) for image in images]