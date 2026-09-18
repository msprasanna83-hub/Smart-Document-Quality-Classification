import cv2
import numpy as np

def validate_document(image_input):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input
        
    if img is None:
        return {"is_document": False, "confidence": 0.0, "reason": "Failed to load image"}
        
    # Resize for consistent processing
    h, w = img.shape[:2]
    target_w = 800
    if w > target_w:
        ratio = target_w / float(w)
        img = cv2.resize(img, (target_w, int(h * ratio)))
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_area = gray.shape[0] * gray.shape[1]
    
    # 1. Edge & Contour Detection for Document Boundaries
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 0.15 * img_area: 
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w_box, h_box = cv2.boundingRect(approx)
                aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
                if 0.3 <= aspect_ratio <= 3.0:
                    return {
                        "is_document": True,
                        "confidence": 0.95,
                        "reason": "Large document-like quadrilateral boundary detected"
                    }

    # 2. Background Uniformity Check (For scanned or close-up documents)
    # Documents typically have a dominant background color (the paper) that covers most of the image.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    peak_val = np.argmax(hist)
    
    # Count pixels within +/- 15 intensity of the dominant peak
    lower = max(0, peak_val - 15)
    upper = min(255, peak_val + 15)
    bg_pixels = np.sum(hist[lower:upper+1])
    bg_ratio = bg_pixels / img_area
    
    if bg_ratio >= 0.50:
        # It has a highly uniform background, characteristic of documents.
        confidence = min(0.6 + bg_ratio * 0.3, 0.95)
        return {
            "is_document": True,
            "confidence": round(float(confidence), 2),
            "reason": f"Highly uniform background detected (Ratio: {bg_ratio:.2f})"
        }
        
    return {
        "is_document": False,
        "confidence": round(float(1.0 - bg_ratio), 2),
        "reason": f"No document boundary and non-uniform background (Ratio: {bg_ratio:.2f})"
    }

if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) > 1:
        print(json.dumps(validate_document(sys.argv[1]), indent=2))
