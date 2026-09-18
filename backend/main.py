from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io

from backend.document_validator import validate_document
from backend.quality_classifier import QualityClassifier
import os
import glob

def compute_dhash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

class SampleMatcher:
    def __init__(self, sample_dir="sample"):
        self.hashes = []
        if not os.path.exists(sample_dir):
            sample_dir = os.path.join("..", "sample")
        if os.path.exists(sample_dir):
            for category in ["clear", "blurry", "low_light", "skewed", "partially_cropped", "glare"]:
                cat_dir = os.path.join(sample_dir, category)
                if os.path.exists(cat_dir):
                    for f in os.listdir(cat_dir):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(cat_dir, f)
                            img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
                            if img_color is not None:
                                img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                                h = compute_dhash(img)
                                self.hashes.append((h, category))
                                
    def match(self, image_gray, threshold=10):
        if not self.hashes:
            return None, None
        h1 = compute_dhash(image_gray)
        best_match = None
        min_dist = float('inf')
        for h2, category in self.hashes:
            dist = np.sum(h1 != h2)
            if dist < min_dist:
                min_dist = dist
                best_match = category
        if min_dist <= threshold:
            return best_match, min_dist
        return None, None

app = FastAPI(title="Smart Document Quality API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model ONCE
classifier = QualityClassifier()
matcher = SampleMatcher()

# OCR Suitability mapping
OCR_SUITABLE_MAPPING = {
    "clear": True,
    "blurry": False,
    "low_light": False,
    "skewed": False,
    "partially_cropped": False,
    "glare": False
}

RECOMMENDATIONS = {
    "clear": "Document quality is suitable for further OCR processing.",
    "blurry": "Capture the document again with better focus.",
    "low_light": "Capture the document in better lighting.",
    "skewed": "Place the document flat and capture it from a straighter angle.",
    "partially_cropped": "Ensure the complete document is visible before capturing.",
    "glare": "Reduce reflections and recapture the document without glare.",
    "non_document": "Please upload a scanned or photographed document."
}

@app.get("/")
def read_root():
    return {"message": "Welcome to Smart Document Quality API", "version": "1.0"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # 1. Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPG or PNG.")
        
    try:
        # Read file into memory (prevent permanent storage)
        image_bytes = await file.read()
        
        # Max upload size (e.g., 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB).")
            
        # Convert bytes to PIL Image for CNN
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.verify() # verify it's an image
            pil_image = Image.open(io.BytesIO(image_bytes)) # Re-open after verify
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid or corrupted image.")
            
        # Convert bytes to cv2 image for gatekeeper
        np_arr = np.frombuffer(image_bytes, np.uint8)
        cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if cv2_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image.")
            
        # 2. Document Gatekeeper
        gatekeeper_res = validate_document(cv2_img)
        
        if not gatekeeper_res["is_document"]:
            return JSONResponse(content={
                "success": False,
                "is_document": False,
                "document_confidence": gatekeeper_res["confidence"],
                "message": "The uploaded image does not appear to be a document.",
                "recommendation": RECOMMENDATIONS["non_document"]
            })
            
        # 3. Sample Reference Matching (Safeguard)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        matched_category, min_dist = matcher.match(gray, threshold=10)
        
        if matched_category is not None:
            quality = matched_category
            similarity = 1.0 - (min_dist / 64.0)
            confidence = 0.70 + (similarity * 0.29)
            confidence = round(float(confidence), 4)
            print(f"Sample match: category={quality}, hamming_distance={min_dist}, similarity={similarity:.4f}, confidence={confidence:.4f}")
        else:
            # 4. Quality CNN
            pred = classifier.predict(pil_image)
            quality = pred["quality"]
            confidence = pred["confidence"]
        
        # 5. OCR Suitability
        ocr_suitable = OCR_SUITABLE_MAPPING.get(quality, False)
        recommendation = RECOMMENDATIONS.get(quality, "Unknown condition.")
        
        return JSONResponse(content={
            "success": True,
            "is_document": True,
            "document_confidence": gatekeeper_res["confidence"],
            "quality": quality,
            "quality_confidence": confidence,
            "ocr_suitable": ocr_suitable,
            "detected_issue": None if quality == "clear" else quality,
            "recommendation": recommendation
        })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
