# Technical Design Document
## Multi-Disease Chest X-Ray Diagnosis System

**Project Name:** Automated Pneumonia Detection from Chest Radiographs  
**Author:** [Your Name]  
**Date:** February 2026  
**Status:** Design Phase  
**Reviewed By:** Senior ML Engineer

---

## 1. Executive Summary

### 1.1 Problem Statement

Manual interpretation of chest X-rays is time-intensive, requiring specialized radiologists, and is subject to inter-observer variability (85-95% diagnostic accuracy). In resource-constrained settings or high-volume scenarios, diagnostic delays can lead to adverse patient outcomes, particularly for time-sensitive conditions like pneumonia.

### 1.2 Proposed Solution

We propose a deep learning-based computer-aided diagnosis (CAD) system that:
- Automatically classifies chest X-rays as Normal or Pneumonia
- Provides visual explanations (GradCAM heatmaps) for clinical interpretability
- Achieves ≥90% AUC-ROC with high sensitivity (recall ≥0.90) to minimize false negatives
- Delivers predictions in <2 seconds via REST API

### 1.3 Business Impact

**Quantified Benefits:**
- Reduce diagnosis time: 15 minutes → 2 seconds (450× faster)
- Scale to unlimited volume (vs. limited radiologist availability)
- Reduce false negative rate through high-recall optimization
- Support triage in emergency departments and low-resource settings

**Stakeholders:**
- **Radiologists:** AI-assisted second opinion, workload reduction
- **Emergency Physicians:** Rapid triage decision support
- **Hospitals:** Improved throughput, reduced diagnostic errors
- **Patients:** Faster diagnosis, earlier treatment initiation

---

## 2. Technical Approach

### 2.1 Why Deep Learning?

**Traditional ML Limitations:**

| Approach | Feature Engineering | Performance on Images | Scalability |
|----------|---------------------|----------------------|-------------|
| SVM/Random Forest | Manual (SIFT, HOG, texture) | Poor (max ~75% accuracy) | Low |
| Classical CNNs | Automatic | Good (80-85% accuracy) | Medium |
| **Transfer Learning CNNs** | **Automatic + Pretrained** | **Excellent (90-95% AUC)** | **High** |

**Why CNNs?**
- Automatic hierarchical feature learning (edges → shapes → disease patterns)
- Parameter sharing reduces overfitting (320 params/filter vs 12.8M for fully connected)
- Spatial invariance through pooling (detects pneumonia anywhere in lung field)

### 2.2 Transfer Learning Justification

**Challenge:** Limited dataset size (~5,000 images) insufficient for training deep networks from scratch (requires ~100K+ images).

**Solution:** Transfer learning from ImageNet-pretrained models.

**Rationale:**
- **Low-level features generalize:** Edge, texture, and gradient detectors learned on natural images (ImageNet) transfer to medical images
- **Fine-tuning strategy:** Freeze early layers (universal features), train middle/late layers on domain-specific patterns
- **Proven efficacy:** CheXNet (Rajpurkar et al.) achieved radiologist-level performance using pretrained DenseNet-121 on 112K chest X-rays

**Expected Improvement:**
- From scratch: ~70% accuracy (underfitting due to data scarcity)
- Transfer learning: ~90% AUC (leverages 1.2M ImageNet pretraining)

---

## 3. Model Architecture Design

### 3.1 Architecture Selection Criteria

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| Medical imaging validation | High | Must be proven on radiology tasks |
| Parameter efficiency | High | Deployment on edge devices, fast inference |
| Interpretability | Critical | Clinical adoption requires explainability |
| Training stability | Medium | Limited compute resources |

### 3.2 Selected Architectures

We will implement and compare **TWO** architectures:

---

#### **Model 1: ResNet50 (Baseline)**

**Architecture:** Residual Network with 50 layers

**Innovation:** Skip connections (residual learning)
```
y = F(x) + x
where F(x) = stacked Conv-BN-ReLU blocks
```

**Advantages:**
- **Solves vanishing gradients:** Skip connections enable gradient flow to early layers
- **Proven medical imaging track record:** Used in 78% of radiology AI papers (2020-2023)
- **Stable training:** Residual learning easier to optimize than plain deep networks
- **Moderate size:** 25.6M parameters (deployable on standard GPUs)

**Architecture Details:**
```
Input: 224×224×1 (grayscale X-ray, resized)
├─ Conv1: 7×7, 64 filters, stride 2
├─ MaxPool: 3×3, stride 2
├─ Residual Blocks:
│   ├─ Stage 1: 3 blocks × [1×1, 64] → [3×3, 64] → [1×1, 256]
│   ├─ Stage 2: 4 blocks × [1×1, 128] → [3×3, 128] → [1×1, 512]
│   ├─ Stage 3: 6 blocks × [1×1, 256] → [3×3, 256] → [1×1, 1024]
│   └─ Stage 4: 3 blocks × [1×1, 512] → [3×3, 512] → [1×1, 2048]
├─ GlobalAveragePooling
├─ Dropout(0.5)
└─ Dense(1, activation='sigmoid')  # Binary classification

Total parameters: 25.6M
Trainable (after freezing first 40 layers): ~5M
```

**Expected Performance:**
- AUC-ROC: 0.88-0.92
- Inference time: 1.8 seconds (single image, CPU)
- Training time: 2-3 hours (50 epochs, GPU)

---

#### **Model 2: EfficientNet-B4 (Primary Model)**

**Architecture:** Compound-scaled ConvNet

**Innovation:** Balanced scaling of depth, width, and resolution
```
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

where α·β²·γ² ≈ 2, φ = compound coefficient
```

**Advantages:**
- **Superior parameter efficiency:** 19M params with higher accuracy than ResNet50 (25M)
- **Compound scaling:** Optimal balance prevents over-parameterization in one dimension
- **Mobile-friendly:** Designed for deployment constraints (EfficientNet-B0 runs on mobile)
- **State-of-art ImageNet:** 83% top-1 accuracy (vs ResNet50's 76%)

**Architecture Details:**
```
Input: 380×380×1 (higher resolution for B4)
├─ Stem: Conv 3×3, 48 filters
├─ MBConv Blocks (Mobile Inverted Bottleneck):
│   ├─ Stage 1: 2 blocks, expansion ratio 1, 24 filters
│   ├─ Stage 2: 4 blocks, expansion ratio 6, 32 filters
│   ├─ Stage 3: 4 blocks, expansion ratio 6, 56 filters
│   ├─ Stage 4: 6 blocks, expansion ratio 6, 112 filters
│   ├─ Stage 5: 6 blocks, expansion ratio 6, 160 filters
│   ├─ Stage 6: 8 blocks, expansion ratio 6, 272 filters
│   └─ Stage 7: 2 blocks, expansion ratio 6, 448 filters
├─ Head: Conv 1×1, 1792 filters
├─ GlobalAveragePooling
├─ Dropout(0.4)
└─ Dense(1, activation='sigmoid')

Total parameters: 19M
Trainable (after freezing first 70%): ~6M
```

**Expected Performance:**
- AUC-ROC: 0.90-0.94
- Inference time: 2.1 seconds (higher resolution overhead)
- Training time: 3-4 hours (50 epochs, GPU)

---

### 3.3 Architecture Comparison

| Metric | ResNet50 | EfficientNet-B4 | Winner |
|--------|----------|-----------------|--------|
| **Parameters** | 25.6M | 19M | EfficientNet ✓ |
| **Input size** | 224×224 | 380×380 | ResNet (faster) |
| **ImageNet Top-1** | 76.2% | 83.0% | EfficientNet ✓ |
| **Medical imaging papers** | 450+ | 120+ | ResNet (more validation) |
| **Training stability** | High | Medium | ResNet ✓ |
| **Parameter efficiency** | 1× baseline | 1.35× fewer params | EfficientNet ✓ |
| **Inference speed (CPU)** | 1.8s | 2.1s | ResNet ✓ |
| **Expected AUC** | 0.88-0.92 | 0.90-0.94 | EfficientNet ✓ |

**Decision:** Train BOTH, compare empirically, potentially ensemble.

---

### 3.4 Why NOT Other Architectures?

| Architecture | Reason for Exclusion |
|--------------|---------------------|
| **VGG16/19** | 138M parameters (too large), no skip connections (poor gradient flow) |
| **Inception-v3** | Complex multi-branch architecture, harder to fine-tune |
| **DenseNet-121** | Considered, but EfficientNet offers better efficiency |
| **Vision Transformer (ViT)** | Requires >100K images for effective training; our 5K dataset insufficient |
| **EfficientNet-B0** | Too small (5M params), may underfit subtle pneumonia patterns |
| **EfficientNet-B7** | Too large (66M params), overfitting risk on 5K images |

---

## 4. Data Strategy

### 4.1 Dataset Selection

**Primary Dataset:** Kaggle Chest X-Ray Images (Pneumonia)
- **Size:** 5,863 images (JPEG)
- **Classes:** Normal (1,583), Pneumonia (4,273)
- **Source:** Pediatric patients (1-5 years), Guangzhou Women and Children's Medical Center
- **Format:** Grayscale, variable dimensions (384-2916 pixels)
- **Labels:** Binary classification

**Class Distribution:**
- Normal: 27%
- Pneumonia: 73%
- **Imbalance ratio:** 2.7:1 (moderate imbalance, requires handling)

### 4.2 Data Preprocessing Pipeline
```python
Preprocessing steps:
1. Resize: 224×224 (ResNet) / 380×380 (EfficientNet)
2. Normalize: mean=0.485, std=0.229 (ImageNet statistics)
   Rationale: Align with pretraining distribution
3. Convert to tensor: [0, 255] → [0, 1] → standardize
4. Channel expansion: 1-channel → 3-channel (repeat grayscale)
   Rationale: Match ImageNet RGB input format
```

### 4.3 Data Augmentation (Training Only)

**Augmentation strategy:**

| Transform | Parameters | Justification |
|-----------|-----------|---------------|
| **RandomRotation** | ±10° | Simulate patient positioning variability |
| **RandomHorizontalFlip** | p=0.5 | Anatomically valid (left/right lung symmetry) |
| **RandomAffine** | translate=0.1 | Account for X-ray alignment variations |
| **RandomResizedCrop** | scale=(0.9, 1.1) | Handle different patient sizes |
| **ColorJitter** | brightness=0.2 | Simulate exposure variations |
| **GaussianBlur** | σ=0.5 | Reduce noise sensitivity |

**NOT used:**
- ❌ Vertical flip (anatomically invalid - heart position)
- ❌ Extreme rotations (>15°) (unrealistic X-ray orientation)
- ❌ Cutout/erasing (may remove critical pathology)

**Expected impact:** +3-5% AUC improvement through regularization

### 4.4 Train/Validation/Test Split
```
Total: 5,863 images

Train: 70% (4,104 images)
├─ Normal: 1,108
└─ Pneumonia: 2,996

Validation: 15% (879 images)
├─ Normal: 237
└─ Pneumonia: 642

Test: 15% (880 images)
├─ Normal: 238
└─ Pneumonia: 642

Stratified split to maintain class distribution
Random seed: 42 (reproducibility)
```

### 4.5 Class Imbalance Mitigation

**Problem:** 73% pneumonia vs 27% normal → model bias toward majority class

**Solution:** Multi-strategy approach

| Technique | Implementation | Expected Impact |
|-----------|---------------|-----------------|
| **Weighted Loss** | `weight = [2.7, 1.0]` for [Normal, Pneumonia] | Penalize Normal misclassification 2.7× |
| **Focal Loss** | `γ=2, α=0.25` | Focus on hard-to-classify examples |
| **Balanced Sampling** | Sample equally from each class per batch | Force model to see balanced data |

**Chosen approach:** Weighted Binary Cross-Entropy (simplest, proven effective)
```python
pos_weight = torch.tensor([num_normal / num_pneumonia])  # 1583/4273 ≈ 0.37
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## 5. Training Strategy

### 5.1 Transfer Learning Approach

**Fine-tuning strategy:**
```
Phase 1: Feature Extraction (Epochs 1-10)
├─ Freeze: All pretrained layers
├─ Train: Only new classification head
├─ Learning rate: 1e-3
└─ Purpose: Adapt head to medical domain

Phase 2: Fine-tuning (Epochs 11-50)
├─ Freeze: First 60% of layers (low-level features)
├─ Train: Last 40% + classification head
├─ Learning rate: 1e-4 (lower to prevent catastrophic forgetting)
└─ Purpose: Adapt mid/high-level features to X-rays
```

**Rationale:**
- Early layers (edges, textures) transfer well → freeze
- Late layers (ImageNet-specific objects) need retraining → unfreeze

### 5.2 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Optimizer** | Adam | Adaptive learning rate, robust to hyperparameter choices |
| **Learning rate** | 1e-4 (fine-tuning) | Small LR prevents overwriting pretrained weights |
| **Batch size** | 32 | Balance: GPU memory (12GB) vs training stability |
| **Epochs** | 50 | With early stopping (patience=10) |
| **Weight decay** | 1e-5 | L2 regularization to prevent overfitting |
| **Dropout** | 0.5 (ResNet), 0.4 (EfficientNet) | Standard regularization |
| **LR scheduler** | ReduceLROnPlateau | Reduce LR by 0.5× if val_loss plateaus for 5 epochs |

### 5.3 Training Procedure
```python
for epoch in range(50):
    # Training phase
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss, val_auc = evaluate(model, val_loader)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print("Early stopping triggered")
            break
```

### 5.4 Regularization Techniques

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| **Dropout** | p=0.5 after GAP layer | Prevent co-adaptation of neurons |
| **Weight Decay** | λ=1e-5 in Adam | L2 regularization on weights |
| **Data Augmentation** | See Section 4.3 | Increase effective dataset size |
| **Early Stopping** | patience=10 epochs | Stop when overfitting detected |
| **Batch Normalization** | Pretrained (frozen) | Stabilize activations |

---

## 6. Evaluation Strategy

### 6.1 Why Not Just Accuracy?

**Problem:** With 73% pneumonia, predicting "all pneumonia" gives 73% accuracy!

**Accuracy is misleading for:**
- Imbalanced datasets
- Medical applications (cost of errors differs)

### 6.2 Primary Metrics

| Metric | Formula | Target | Justification |
|--------|---------|--------|---------------|
| **AUC-ROC** | Area under ROC curve | ≥0.90 | Threshold-independent, handles imbalance |
| **Recall (Sensitivity)** | TP / (TP + FN) | ≥0.90 | **CRITICAL:** Cannot miss pneumonia (FN = patient dies) |
| **Precision** | TP / (TP + FP) | ≥0.85 | Minimize false alarms (FP = unnecessary treatment) |
| **F1-Score** | 2 × (P × R) / (P + R) | ≥0.87 | Balanced performance |

### 6.3 Secondary Metrics

- **Specificity:** TN / (TN + FP) ≥0.80
- **Average Precision (AP):** Precision-recall curve area
- **Confusion Matrix:** Detailed error analysis
- **Per-class AUC:** Separate for Normal and Pneumonia

### 6.4 Clinical Performance Benchmarks

| Benchmark | Performance | Source |
|-----------|-------------|--------|
| Human radiologists | 85-95% accuracy | Literature review |
| CheXNet (Stanford) | 0.94 AUC | Rajpurkar et al. 2017 |
| Our target | **0.90 AUC** | Conservative, achievable |

### 6.5 Error Analysis

**Mandatory post-training analysis:**

1. **False Negative Analysis (HIGH PRIORITY)**
   - Which pneumonia cases were missed?
   - Common visual characteristics?
   - Are they subtle/early-stage cases?

2. **False Positive Analysis**
   - Which normal cases flagged as pneumonia?
   - Presence of confounders (medical devices, artifacts)?

3. **Confidence Calibration**
   - Are high-confidence predictions more accurate?
   - Set decision threshold (default 0.5 may not be optimal)

---

## 7. Model Interpretability

### 7.1 Clinical Necessity

**Why explainability is NON-NEGOTIABLE:**
- Doctors won't trust black-box predictions
- Regulatory requirements (FDA, CE marking)
- Debugging model errors
- Educational tool for medical students

### 7.2 GradCAM Implementation

**Gradient-weighted Class Activation Mapping**

**How it works:**
```
1. Forward pass → get prediction
2. Compute gradient of predicted class w.r.t. last conv layer
3. Global average pooling of gradients → weights
4. Weighted combination of feature maps
5. ReLU (keep positive contributions)
6. Upsample to input size → heatmap overlay
```

**Implementation:**
```python
from pytorch_grad_cam import GradCAM

# Target layer: last convolutional layer
target_layers = [model.layer4[-1]]  # ResNet50
# target_layers = [model.blocks[-1]]  # EfficientNet

cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor, target_category=1)  # Pneumonia class

# Overlay heatmap on X-ray
visualization = overlay_heatmap(xray_image, grayscale_cam)
```

**Output:** 
- Heatmap highlighting regions contributing to "Pneumonia" prediction
- **Clinical validation:** Should highlight lung opacities, not ribs/heart

### 7.3 Validation of Interpretability

**Quality checks:**

| Check | Pass Criteria | Action if Fail |
|-------|---------------|----------------|
| Heatmap focuses on lungs | ≥80% of high-attention pixels in lung region | Retrain or investigate |
| Ignores non-pathological structures | Low attention on ribs, heart borders | Acceptable |
| Highlights known opacities | Visual correlation with ground truth | Validate with radiologist |

---

## 8. Ensemble Strategy

### 8.1 Rationale

**Why ensemble?**
- Single models have variance (different architectures, different errors)
- Ensemble reduces variance → more robust predictions
- Standard practice in medical AI competitions (Kaggle, RSNA)

### 8.2 Ensemble Method

**Soft Voting (Probability Averaging):**
```python
# Predictions from both models
prob_resnet = resnet_model.predict(xray)      # e.g., 0.85
prob_efficientnet = efficientnet_model.predict(xray)  # e.g., 0.92

# Ensemble prediction
ensemble_prob = 0.5 * prob_resnet + 0.5 * prob_efficientnet
# Result: 0.885
```

**Alternative: Weighted Voting**
```python
# Weight by validation AUC
w_resnet = 0.91 / (0.91 + 0.93)  # ≈ 0.49
w_efficientnet = 0.93 / (0.91 + 0.93)  # ≈ 0.51

ensemble_prob = w_resnet * prob_resnet + w_efficientnet * prob_efficientnet
```

### 8.3 Expected Improvement

**Literature benchmarks:**
- Single best model: 0.92 AUC
- Ensemble of 2-3 models: 0.94-0.95 AUC
- **Expected gain:** +2-3% AUC

---

## 9. Deployment Architecture

### 9.1 System Design
```
Client (Web/Mobile)
    ↓
    HTTP Request (POST /predict)
    ↓
Load Balancer (Nginx)
    ↓
FastAPI Server (Python 3.8)
├─ Input validation
├─ Preprocessing
├─ Model inference (ONNX Runtime)
├─ GradCAM generation
└─ Response formatting
    ↓
    JSON Response
    {
      "prediction": "Pneumonia",
      "confidence": 0.94,
      "gradcam_url": "https://...",
      "inference_time_ms": 1850
    }
```

### 9.2 Model Serving

**Technology:** ONNX (Open Neural Network Exchange)

**Why ONNX?**
- Framework-agnostic (PyTorch → ONNX → any runtime)
- Optimized inference (2-3× faster than PyTorch)
- Smaller model size (FP32 → FP16 quantization)

**Conversion:**
```python
# PyTorch → ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet50_pneumonia.onnx")

# ONNX Runtime inference
import onnxruntime as ort
session = ort.InferenceSession("resnet50_pneumonia.onnx")
output = session.run(None, {"input": xray_array})
```

### 9.3 API Specification

**Endpoint:** `POST /api/v1/predict`

**Request:**
```json
{
  "image": "<base64-encoded X-ray>",
  "return_gradcam": true,
  "model": "ensemble"  // Options: "resnet50", "efficientnet", "ensemble"
}
```

**Response:**
```json
{
  "prediction": {
    "class": "Pneumonia",
    "probability": 0.94,
    "threshold": 0.5
  },
  "gradcam": {
    "heatmap_base64": "<base64-encoded heatmap>",
    "top_regions": ["right lower lobe", "left middle lobe"]
  },
  "metadata": {
    "model_version": "v1.2.0",
    "inference_time_ms": 1850,
    "timestamp": "2026-02-03T14:32:10Z"
  }
}
```

### 9.4 Performance Requirements

| Metric | Target | Monitoring |
|--------|--------|------------|
| **Inference latency (p95)** | <2.5 seconds | Prometheus + Grafana |
| **Throughput** | 100 requests/min | Load testing |
| **Model accuracy** | AUC ≥0.90 | Weekly evaluation on hold-out set |
| **Uptime** | 99% | Health check endpoint |

### 9.5 Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY src/ src/
COPY deployment/api/app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/ensemble.onnx
    volumes:
      - ./models:/app/models
    restart: always
```

---

## 10. Experiment Tracking & MLOps

### 10.1 MLflow Integration

**Track for each experiment:**
- Hyperparameters (LR, batch size, dropout)
- Metrics (train/val loss, AUC, recall, precision)
- Artifacts (model checkpoints, plots, GradCAM examples)
- Code version (Git commit hash)

**Example logging:**
```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({
        "model": "ResNet50",
        "lr": 1e-4,
        "batch_size": 32,
        "augmentation": "standard"
    })
    
    mlflow.log_metrics({
        "train_loss": 0.32,
        "val_auc": 0.91,
        "val_recall": 0.93
    })
    
    mlflow.pytorch.log_model(model, "model")
```

### 10.2 Model Versioning

**Semantic versioning:** v{major}.{minor}.{patch}
- **Major:** Architecture change (ResNet → EfficientNet)
- **Minor:** Significant performance improvement (AUC +0.05)
- **Patch:** Bug fixes, minor retraining

**Model registry:**
```
models/
├─ resnet50_v1.0.0_auc0.89.pth
├─ resnet50_v1.1.0_auc0.91.pth
├─ efficientnet_v1.0.0_auc0.92.pth
└─ ensemble_v2.0.0_auc0.94.onnx  ← Production
```

---

## 11. Success Criteria

### 11.1 Technical Metrics (Must Achieve)

| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| AUC-ROC | ≥0.90 | 0.88 |
| Recall (Pneumonia) | ≥0.90 | 0.85 |
| Precision | ≥0.85 | 0.80 |
| F1-Score | ≥0.87 | 0.83 |
| Inference time | <2.5s | 3.0s |
| Model size | <500MB | 1GB |

### 11.2 Deliverables Checklist

- [ ] Trained ResNet50 model (AUC ≥0.88)
- [ ] Trained EfficientNet-B4 model (AUC ≥0.90)
- [ ] Ensemble model (AUC ≥0.92)
- [ ] GradCAM visualization for all predictions
- [ ] REST API with <2.5s latency
- [ ] Streamlit demo UI
- [ ] Docker container
- [ ] Comprehensive documentation
- [ ] GitHub repository with clean code
- [ ] Model performance report
- [ ] Presentation deck
- [ ] Demo video

### 11.3 Project Timeline

| Phase | Duration | Key Milestone |
|-------|----------|---------------|
| Setup & Documentation | Days 0-2 | ✅ Technical design approved |
| Data preparation | Day 3 | ✅ EDA complete, pipeline ready |
| Baseline model | Day 4 | ✅ ResNet50 trained (AUC ≥0.85) |
| Primary model | Days 5-6 | ✅ EfficientNet-B4 trained (AUC ≥0.90) |
| Ensemble & interpretability | Days 7-9 | ✅ GradCAM working, ensemble AUC ≥0.92 |
| Deployment | Days 10-12 | ✅ API + UI functional |
| Testing & polish | Days 13-15 | ✅ All deliverables complete |

---

## 12. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model underperforms (<0.85 AUC) | Medium | High | Try DenseNet-121, increase augmentation, ensemble |
| Overfitting on small dataset | High | Medium | Strong regularization, early stopping, cross-validation |
| GradCAM highlights wrong regions | Low | High | Validate with radiologist, try GradCAM++ |
| Inference too slow (>3s) | Low | Medium | ONNX quantization, smaller input size, GPU inference |
| Cannot deploy Docker | Low | Low | Provide local deployment script |

---

## 13. Future Enhancements (Post-MVP)

1. **Multi-label classification:** Extend to 14 diseases (NIH dataset)
2. **Attention mechanisms:** Implement CBAM or SE blocks
3. **Test-time augmentation:** Average predictions over multiple augmented inputs
4. **Uncertainty quantification:** Monte Carlo dropout for confidence intervals
5. **Cloud deployment:** AWS SageMaker or Google AI Platform
6. **CI/CD pipeline:** GitHub Actions for automated testing and deployment
7. **A/B testing:** Compare model versions in production
8. **Feedback loop:** Collect physician corrections to retrain model

---

## 14. References

1. Rajpurkar et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays
2. He et al. (2016). Deep Residual Learning for Image Recognition
3. Tan & Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
4. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
5. Kaggle Chest X-Ray Images (Pneumonia) Dataset

---

**Document Status:** ✅ APPROVED FOR IMPLEMENTATION  
**Next Steps:** Begin data acquisition and preprocessing (Day 3)