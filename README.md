# Rice Leaf Disease Classifier

A practice project using transfer learning to classify rice leaf diseases from photos. Uses EfficientNet-B0 or ResNet-50 with a Streamlit web interface. Supports Khmer and English languages.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Upload a rice leaf image to see predictions

## What It Does

Classifies rice leaves into 6 categories:
- Healthy
- Bacterial leaf blight
- Brown spot
- Leaf blast
- Leaf scald
- Sheath blight

The model shows confidence scores for each prediction and includes basic management advice for each disease.

## How It Works

- **Model**: Transfer learning with pre-trained EfficientNet-B0 or ResNet-50
- **Training**: Standard approach with data augmentation and train/val split
- **Interface**: Streamlit app with bilingual support (Khmer/English)
- **Dataset**: Rice disease images from Kaggle

## Project Structure

```
KhmerRice-Net/
├── app.py                    # Streamlit web interface
├── requirements.txt          # Dependencies
├── src/
│   ├── model_def.py          # CNN model definition
│   ├── train_rice.py         # Training script
│   ├── inference.py          # Model loading & prediction
│   ├── preprocess.py         # Image preprocessing
│   ├── rice_dataset.py       # PyTorch dataset class
│   ├── prepare_rice_kaggle.py # Dataset downloader
│   ├── models/
│   │   └── rice_cnn_model.pth # Trained model
│   └── reports/
│       ├── cnn_architecture.txt
│       └── model_summary.txt
└── utils/
    ├── label_map.py          # Disease labels (Khmer/English)
    └── folder_map.py         # Folder mappings
```

## Setup

**Prerequisites**: Python 3.8+

```bash
# Clone and setup
git clone https://github.com/vibecoder1998/KhmerRice-Net.git
cd KhmerRice-Net
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Download dataset
python src/prepare_rice_kaggle.py
```

## Usage

**Running the app**:
```bash
streamlit run app.py
```
Then open `http://localhost:8501` and upload a rice leaf image.

**Training a model**:
```bash
python src/train_rice.py --backbone efficientnet_b0 --epochs 12
```

Options: `--data` (dataset path), `--backbone` (efficientnet_b0 or resnet50), `--epochs`

**Programmatic use**:
```python
from src.inference import load_model_and_metadata, predict
from PIL import Image

model, classes, _ = load_model_and_metadata()
image = Image.open("leaf.jpg").convert("RGB")
label, confidence, probs = predict(model, image, classes)
print(f"{label}: {confidence*100:.1f}%")
```

## How It Works

- **Model**: Transfer learning with EfficientNet-B0 or ResNet-50 backbone
- **Training**: Two-stage approach (freeze backbone, then fine-tune)
- **Data**: ~3,800 images from Kaggle (6 disease classes, well-balanced)
- **Input**: 224×224 RGB images
- **Output**: 6 classifications (healthy + 5 diseases)

**Diseases classified**:
- Healthy
- Bacterial leaf blight
- Brown spot
- Leaf blast
- Leaf scald
- Sheath blight

Labels are provided in both Khmer (ខ្មែរ) and English with basic management notes for each.

## Files

- **`model_def.py`**: RiceCNN class, supports EfficientNet-B0 (~25MB) or ResNet-50 (~100MB)
- **`train_rice.py`**: Training script with standard PyTorch patterns (Adam optimizer, cross-entropy loss, 20% validation split)
- **`inference.py`**: Model inference helpers
- **`preprocess.py`**: Image resizing, normalization, tensor conversion
- **`label_map.py`**: Disease names and basic field management advice per disease
- **`app.py`**: Streamlit UI with upload, confidence display, disease info

## Notes

- Model runs on CPU or GPU (auto-detects CUDA availability)
- Inference is fast enough for real-time use
- The dataset is balanced across 6 classes (~630 images each)
- Data augmentation includes rotation, flip, color jitter during training
- Confidence scores are useful but not always reliable on out-of-distribution images
