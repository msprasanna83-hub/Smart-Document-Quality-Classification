import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import argparse

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

def predict_image(image_path, model_path="models/best_model.pth", class_map_path="models/class_map.json"):
    if not os.path.exists(image_path):
        return {"error": "Image not found"}
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
    inv_class_map = {v: k for k, v in class_map.items()}
    
    model = CustomCNN(num_classes=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    prob_np = probs.cpu().numpy()
    pred_idx = int(torch.argmax(probs).item())
    pred_class = inv_class_map[pred_idx]
    
    probabilities = {inv_class_map[i]: float(prob_np[i]) for i in range(6)}
    
    result = {
        "quality": pred_class,
        "confidence": round(float(prob_np[pred_idx]) * 100, 2),
        "probabilities": probabilities
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help="Path to the image to predict")
    args = parser.parse_args()
    
    res = predict_image(args.image_path)
    print(json.dumps(res, indent=2))
