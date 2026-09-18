# Smart Document Quality Classification

## Problem
Classify document image quality before OCR processing to ensure high OCR accuracy and provide users with actionable feedback to recapture poor-quality documents.

## Six Quality Classes
The model evaluates images into one of six classes:
- **clear**: Perfect for OCR
- **blurry**: Out of focus or motion blur
- **low_light**: Underexposed or dark
- **skewed**: Photographed at an angle / perspective distortion
- **partially_cropped**: Essential parts of the document are cut off
- **glare**: Strong localized reflection or overexposure

## Model Architecture
Custom Convolutional Neural Network (CNN) built and trained from scratch using PyTorch. 
- Input: RGB document image resized to 224×224.
- Output: Quality class, confidence score, detected issue, and OCR suitability boolean.

## Document Validation Gatekeeper
Before quality classification, the image passes through an OpenCV-based document gatekeeper. This heuristic layer uses quadrilateral contour detection and background uniformity histograms to ensure the uploaded image is actually a document (rejecting photos of people, landscapes, objects) before passing it to the CNN.

## Important Evaluation Limitation
The primary test dataset achieves an **84.6% test accuracy**, but it relies heavily on synthetic quality transformations applied to flat document scans. This synthetic metric should **NOT** be presented as a guaranteed real-world accuracy rate. 

Real-world validation identified the following difficulties:
1. **partially_cropped**: Difficult to separate from "zoomed in" or "large font" documents. Real-world crops maintain original aspect ratios, whereas synthetic crops were heavily stretched, causing domain shift.
2. **glare**: Real-world glare washes out contrast and reflects off paper texture. Synthesizing glare on perfectly white B&W document scans resulted in spots that were virtually indistinguishable from the baseline white paper.

## Getting Started
1. Run backend: `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`
2. Run frontend: `npm run dev`
3. Open `http://localhost:3000`
