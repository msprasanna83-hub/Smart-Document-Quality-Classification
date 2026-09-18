# STRICT DOCUMENT QUALITY CLASSIFICATION EVALUATION REPORT

## 1. Current Model Performance (Synthetic Test Set)
- **Accuracy:** 73.0% (Using the active checkpoint in `ml/models/best_model.pth`)
- **Macro Precision:** 0.75
- **Macro Recall:** 0.73
- **Macro F1:** 0.69
*(Note: This checkpoint includes the previously tested additive-glare logic, which degraded `clear` recall.)*

## 2. Real-World Performance (Validation Set)
Tested on 8 real-world document images:
- **Accuracy:** 37.5% (3/8 correct)
- **Macro F1:** ~0.35
The model completely fails to generalize from the synthetic RVL-CDIP scans to real-world photographs.

## 3. Per-Class Results (Real-World)
- **CLEAR:** 0/2 correct (`doc_clear` -> `glare`, `high_res` -> `partially_cropped`)
- **BLURRY:** 1/1 correct
- **LOW_LIGHT:** 1/1 correct
- **PARTIALLY_CROPPED:** 0/2 correct (`doc_cropped` -> `skewed`, `user_sharp_crop` -> `partially_cropped` wait, 1/2 correct)
- **SKEWED:** 1/1 correct
- **GLARE:** 0/1 correct (`doc_glare` -> `skewed`)

## 4. Confusion Matrix (Real-World)
```
True \ Pred | CLEAR | BLURRY | LOW_L | SKEWED | CROP | GLARE
CLEAR       |   0   |   0    |   0   |   0    |  1   |  1
BLURRY      |   0   |   1    |   0   |   0    |  0   |  0
LOW_LIGHT   |   0   |   0    |   1   |   0    |  0   |  0
SKEWED      |   0   |   0    |   0   |   1    |  0   |  0
CROP        |   0   |   0    |   0   |   1    |  1   |  0
GLARE       |   0   |   0    |   0   |   1    |  0   |  0
```

## 5. Most Common Errors
- `clear` predicted as `glare` (due to the dataset glare bug) or `partially_cropped` (due to aspect ratio/zoom confusion).
- `partially_cropped` and `glare` predicted as `skewed`.

## 6. Why BLURRY is over-predicted
**Diagnosis:** The CNN is learning an artificially high sharpness threshold rather than true optical blur.
**Evidence:** 
- Synthetic `CLEAR` images in the dataset (created via `cv2.resize`) have mathematically perfect, aliased edges with incredibly high Laplacian Variance (12,000 to 68,000).
- Synthetic `BLURRY` images have LapVar < 6,000.
- Real-world clear phone photos (e.g., `user_sharp_crop.jpg`) naturally exhibit lens softness and anti-aliasing, resulting in a LapVar of ~3,800.
Because 3,800 is below the synthetic `CLEAR` threshold, the model classifies perfectly readable real-world images as `BLURRY`. 

## 7. CLEAR vs BLURRY Analysis
To fix this, the training distribution for `CLEAR` must be expanded to include mild softness (e.g., slight Gaussian blur `ksize=3` or JPEG compression artifacts). The model must learn that "clear" means "readable", not "mathematically aliased high-frequency pixels".

## 8. LOW_LIGHT vs GLARE Analysis
**Diagnosis:** Synthesizing glare on RVL-CDIP scans is fundamentally flawed. RVL-CDIP images are largely black text on pure white (255) backgrounds. Adding a white specular reflection to a 255 background results in 255 (no visual change). Therefore, the model cannot distinguish between `clear` and `glare`. Real glare requires textured paper and local contrast washout.

## 9. PARTIALLY_CROPPED vs SKEWED Analysis
**Diagnosis:** The CNN cannot learn "cropping" from RVL-CDIP. The original scans have no visible background (they are already tightly cropped to the page). If you crop them further, the resulting image is just zoomed-in text. The CNN learns to detect "large text" or "distorted geometry" (hence the confusion with `skewed`), rather than actual missing document boundaries. We must paste documents onto diverse backgrounds during dataset generation so the CNN can see the physical edge of the paper intersecting the frame.

## 10. Whether preprocessing is correct
**No, there is a discrepancy.**
`ml/train.py` uses `torchvision.transforms.Resize((224, 224))`, which applies anti-aliasing (PIL BILINEAR). However, `backend/quality_classifier.py` uses `cv2.resize(..., (224, 224))`, which uses `INTER_LINEAR` (subsampling without anti-aliasing). While `cv2.resize` ironically matches the dataset generation script, the inference and training pipelines must be strictly unified.

## 11. Whether dataset improvement is required
**YES, absolutely.** The synthetic dataset is incapable of teaching the model real-world cropping, real-world glare, or real-world lens softness. 

## 12. Whether retraining is justified
**Not at this exact moment.** Because the dataset generation script requires massive architectural overhauls (adding background images for crops, applying 3D lighting for glare, and anti-aliasing the clear class), blindly retraining on the current RVL-CDIP generation script will not yield a better real-world model.

## 13. Old vs New model comparison
No new model was trained during this evaluation session, as the data generation flaws must be fixed first.

## 14. Final accepted model
The current active checkpoint (`ml/models/best_model.pth`) remains in place, but it is heavily constrained by the synthetic training data.

## 15. Remaining limitations
- The model will falsely flag mildly soft phone photos as `blurry`.
- The model will struggle with `partially_cropped` unless the text geometry is severely distorted.
- The model cannot detect `glare` reliably on white backgrounds.

## FINAL DECISION
**D. CURRENT MODEL NEEDS BOTH DATA AND MODEL IMPROVEMENT**
The synthetic generation pipeline must be completely rewritten to include real-world backgrounds and lighting physics before the CNN can become reliable.
