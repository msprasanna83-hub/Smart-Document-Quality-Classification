import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np
import cv2

class CustomCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class QualityClassifier:
    def __init__(self, model_path="../ml/models/best_model.pth", class_map_path="../ml/models/class_map.json"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # We need absolute path or relative to backend
        # backend is running in CWD (project root) or backend folder?
        # Let's assume CWD is project root when running backend.
        if not os.path.exists(model_path):
            model_path = os.path.join("ml", "models", "best_model.pth")
            class_map_path = os.path.join("ml", "models", "class_map.json")
            
        with open(class_map_path, 'r') as f:
            class_map = json.load(f)
        self.inv_class_map = {v: k for k, v in class_map.items()}
        
        self.model = CustomCNN(num_classes=6)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def predict(self, image: Image.Image):
        # Convert PIL to cv2 image for identical preprocessing to training pipeline
        img_np = np.array(image.convert('RGB'))
        # PIL is RGB, cv2 expects BGR but since we just need it for resizing it doesn't matter
        # as long as we keep track of channels. Let's just resize the RGB numpy array.
        import cv2
        img_resized = cv2.resize(img_np, (224, 224))
        
        # Convert back to PIL for torchvision ToTensor
        img_pil = Image.fromarray(img_resized)
        
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        prob_np = probs.cpu().numpy()
        pred_idx = int(torch.argmax(probs).item())
        pred_class = self.inv_class_map[pred_idx]
        
        probabilities = {self.inv_class_map[i]: float(prob_np[i]) for i in range(6)}
        
        return {
            "quality": pred_class,
            "confidence": round(float(prob_np[pred_idx]), 4),
            "probabilities": probabilities
        }
