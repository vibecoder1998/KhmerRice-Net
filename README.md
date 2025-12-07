# Rice Leaf Disease Classification System 🌾

A deep learning-based system for classifying rice leaf diseases using Convolutional Neural Networks (CNN). This project provides a bilingual interface (Khmer/English) for farmers and agricultural researchers to identify common rice diseases from leaf images.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Supported Diseases](#supported-diseases)
- [File Descriptions](#file-descriptions)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements a neural network-based classification system for recognizing rice leaf diseases. The system can identify six different rice leaf conditions from images, making it useful for:

- **Early Disease Detection** in rice fields
- **Agricultural Research** and monitoring
- **Farmer Education** and disease identification
- **Precision Agriculture** applications
- **Automated Crop Health Monitoring**

The system uses transfer learning with pre-trained models (EfficientNet-B0 or ResNet-50) and provides a user-friendly bilingual web interface for real-time disease classification.

## ✨ Features

- **Multi-class Disease Classification**: Identifies 6 different rice leaf conditions
- **Bilingual Interface**: Supports both Khmer (ខ្មែរ) and English languages
- **Transfer Learning**: Uses EfficientNet-B0 or ResNet-50 pre-trained on ImageNet
- **Streamlit Web App**: User-friendly web interface for disease prediction
- **Real-time Inference**: Fast prediction with confidence scores
- **Actionable Recommendations**: Provides field management advice for each disease
- **Comprehensive Documentation**: Includes model cards, architecture diagrams, and reports

## 📁 Project Structure

```
KhmerRice-Net/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── model_def.py                # Neural network model definition
│   ├── preprocess.py               # Image preprocessing utilities
│   ├── inference.py                # Model loading and prediction logic
│   ├── rice_dataset.py             # PyTorch Dataset class
│   ├── train_rice.py               # Training script
│   ├── prepare_rice_kaggle.py      # Dataset preparation script
│   ├── data/                       # Dataset directory
│   │   └── rice/                   # Rice disease images (6 classes)
│   │       ├── bacterial_leaf_blight/
│   │       ├── brown_spot/
│   │       ├── healthy/
│   │       ├── leaf_blast/
│   │       ├── leaf_scald/
│   │       └── sheath_blight/
│   ├── models/                     # Trained model checkpoints
│   │   └── rice_cnn_model.pth      # Pre-trained CNN model
│   └── reports/                    # Documentation and reports
│       ├── cnn_architecture.txt    # Architecture diagram
│       └── model_summary.txt       # Model specifications
└── utils/                          # Utility modules
    ├── __init__.py
    ├── label_map.py                # Disease label mappings and information
    ├── mapping.py                  # Original label mappings
    └── folder_map.py               # Folder structure mappings
```

## 🏗️ Architecture

### Model Architecture

The project uses **transfer learning** with a pre-trained backbone network and a custom classifier head:

**Backbone Options:**
- **EfficientNet-B0**: ~6M parameters, ~25MB model size
- **ResNet-50**: ~26.6M parameters, ~100MB model size

**Architecture Design:**
```
Input Layer: 224×224×3 RGB images
    ↓
Backbone Network: Pre-trained on ImageNet
    - EfficientNet-B0 or ResNet-50
    - Feature extraction: 1,280 or 2,048 dimensions
    ↓
Classifier Head:
    - Linear Layer: backbone_features → 512
    - ReLU Activation
    - Dropout (0.4) for regularization
    - Linear Layer: 512 → 6 (disease classes)
    ↓
Output: 6-dimensional probability vector
```

### Processing Pipeline

1. **Image Upload**: User uploads an image via Streamlit interface
2. **Preprocessing**:
   - Convert to RGB
   - Resize to 224×224 pixels
   - Normalize with ImageNet statistics
   - Convert to tensor
3. **Model Inference**:
   - Load pre-trained model
   - Forward pass through neural network
   - Apply softmax to get probability distribution
4. **Post-processing**:
   - Extract top prediction and confidence score
   - Map label to Khmer/English names
   - Display results with actionable advice

### Training Strategy

**Two-Stage Training Approach:**

1. **Stage 1: Classifier Training (3 epochs)**
   - Freeze backbone parameters
   - Train only classifier head
   - Learning rate: 1e-3
   - Optimizer: Adam

2. **Stage 2: Fine-tuning (9+ epochs)**
   - Unfreeze all parameters
   - Fine-tune entire model
   - Learning rate: 1e-4 (reduced)
   - Optimizer: Adam

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework for the neural network model
- **Streamlit**: Web framework for the interactive UI
- **PIL (Pillow)**: Image processing and manipulation
- **NumPy**: Numerical operations and array handling
- **scikit-learn**: Data splitting and label encoding utilities
- **torchvision**: Pre-trained models and image transformations
- **Python 3.8+**: Programming language

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- CUDA-capable GPU (optional, for faster training)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vibecoder1998/KhmerRice-Net.git
   cd KhmerRice-Net
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files exist**:
   - Ensure `src/models/rice_cnn_model.pth` exists
   - If not, train a model using the training script (see [Usage](#usage))

5. **Download dataset** (if not already present):
   ```bash
   python src/prepare_rice_kaggle.py
   ```

## 📖 Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**:
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

3. **Use the application**:
   - Click "Upload Image" or drag and drop an image file
   - Supported formats: PNG, JPG, JPEG
   - Wait for the classification results
   - View the predicted disease, confidence score, and top predictions
   - Read actionable recommendations for field management

### Training the Model

Train a new model using the training script:

```bash
python src/train_rice.py --backbone efficientnet_b0 --epochs 12
```

**Arguments:**
- `--data`: Path to dataset directory (default: `data/rice`)
- `--backbone`: Architecture choice - `efficientnet_b0` or `resnet50` (default: `efficientnet_b0`)
- `--epochs`: Total training epochs (default: 12)

**Example:**
```bash
# Train with EfficientNet-B0
python src/train_rice.py --backbone efficientnet_b0 --epochs 15

# Train with ResNet-50
python src/train_rice.py --backbone resnet50 --epochs 12
```

The trained model will be saved to `src/models/rice_cnn_model.pth` (or the path specified in the script).

### Using the Model Programmatically

```python
from src.inference import load_model_and_metadata, predict
from src.preprocess import preprocess_image
from PIL import Image
from utils.label_map import DISEASE_INFO

# Load model and metadata
model, classes, transform_info = load_model_and_metadata()

# Load and preprocess an image
image = Image.open("path/to/rice_leaf.jpg")
image = image.convert("RGB")

# Make prediction
top_label, confidence, all_probs = predict(model, image, classes)

# Get disease information
info = DISEASE_INFO[top_label]["en"]
print(f"Predicted: {info['name']} with {confidence*100:.2f}% confidence")
print("Advice:", info["advice"])
```

## 🤖 Model Details

### Training Specifications

- **Model Type**: Convolutional Neural Network with Transfer Learning
- **Input Size**: 224×224×3 RGB images
- **Output Classes**: 6 rice disease classes
- **Activation Functions**: ReLU for hidden layers, Softmax for output
- **Regularization**: Dropout (0.4) to prevent overfitting
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Batch Size**: 32
- **Validation Split**: 20% (stratified)

### Model Performance

- **Inference Speed**: 
  - CPU: <100ms per image
  - GPU: <10ms per image
- **Memory Usage**:
  - EfficientNet-B0: ~200MB
  - ResNet-50: ~1GB
- **Model Size**:
  - EfficientNet-B0: ~25MB
  - ResNet-50: ~100MB

### Data Augmentation

**Training Augmentations:**
- Random horizontal flip (50% probability)
- Random rotation (±12 degrees)
- Color jitter (brightness, contrast, saturation: 0.3)

**Validation:**
- Only resize and normalization (no augmentation)

## 📊 Dataset

- **Source**: Rice Disease Dataset from Kaggle
- **Total Images**: ~3,829
- **Classes**: 6
- **Split**: 80% train, 20% validation (stratified)
- **Class Distribution**:
  - Bacterial Leaf Blight: 636 images
  - Brown Spot: 646 images
  - Healthy: 653 images
  - Leaf Blast: 634 images
  - Leaf Scald: 628 images
  - Sheath Blight: 632 images

The dataset is well-balanced across classes, with each class containing approximately 630-650 images.

## 🔤 Supported Diseases

The model can classify **6 rice leaf conditions**:

| Disease | Khmer Name | English Name |
|---------|-----------|--------------|
| `bacterial_leaf_blight` | ជម្ងឺខ្លាញ់ស្លឹកបាក់តេរី | Bacterial Leaf Blight (BLB) |
| `brown_spot` | ជម្ងឺចំណុចត្នោត | Brown Spot |
| `healthy` | ស្រូវមានសុខភាពល្អ | Healthy Leaf |
| `leaf_blast` | ជម្ងឺផ្លេះស្លឹក | Leaf Blast |
| `leaf_scald` | ជម្ងឺដុតស្លឹក | Leaf Scald |
| `sheath_blight` | ជម្ងឺខ្លាញ់កណ្ដាលដើម | Sheath Blight |

Each disease classification includes:
- Disease name in Khmer and English
- Confidence score
- Top predictions with probabilities
- Actionable field management recommendations

## 📝 File Descriptions

### Core Application Files

- **`app.py`**: Main Streamlit application with UI components, file upload handling, and result display
- **`requirements.txt`**: Python package dependencies list

### Source Code Modules (`src/`)

- **`model_def.py`**: Defines the `RiceCNN` class with neural network architecture
- **`preprocess.py`**: Contains `preprocess_image()` and `get_transform()` functions for image normalization and tensor conversion
- **`inference.py`**: Provides `load_model_and_metadata()` and `predict()` functions for model inference
- **`rice_dataset.py`**: PyTorch Dataset class for loading and preprocessing training data
- **`train_rice.py`**: Training script with two-stage training approach
- **`prepare_rice_kaggle.py`**: Script to download and prepare dataset from Kaggle

### Utility Modules (`utils/`)

- **`label_map.py`**: Dictionary mapping disease labels to Khmer/English names, disease information, and management advice
- **`mapping.py`**: Original label mappings (kept for backward compatibility)
- **`folder_map.py`**: Mappings for disease folder structures

### Model Files (`src/models/`)

- **`rice_cnn_model.pth`**: Serialized PyTorch model state dictionary with classes and backbone information

## 🔍 API Reference

### `src.model_def.RiceCNN`

```python
class RiceCNN(nn.Module):
    def __init__(self, backbone="efficientnet_b0", num_classes=6, freeze_backbone=True)
    def unfreeze_backbone(self)
    def forward(self, x)
```

### `src.preprocess`

```python
def preprocess_image(image, image_size=224) -> torch.Tensor
def get_transform(image_size=224, augment=False) -> transforms.Compose
```

### `src.inference`

```python
def load_model_and_metadata(checkpoint_path=None) -> tuple
    # Returns: (model, classes, transform_info)

def predict(model, image, classes=None, top_k=3) -> tuple
    # Returns: (top_label, confidence, all_probs)
```

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**:
   - Ensure `src/models/rice_cnn_model.pth` exists
   - Check file paths are correct
   - Train a model if it doesn't exist

2. **Import errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Ensure virtual environment is activated
   - Check Python version (3.8+)
   - Make sure you're running from the project root directory

3. **Low prediction confidence**:
   - Image quality may be poor
   - Disease symptoms may not be clearly visible
   - Try images with clear, well-lit rice leaves
   - Ensure image contains a rice leaf

4. **Streamlit not starting**:
   - Check if port 8501 is available
   - Use `streamlit run app.py --server.port 8502` to use a different port

5. **CUDA/GPU errors**:
   - Model will automatically use CPU if CUDA is not available
   - For GPU training, ensure PyTorch with CUDA support is installed
   - Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

6. **Dataset download issues**:
   - Ensure Kaggle credentials are set up (if using KaggleHub)
   - Check internet connection
   - Verify dataset path in `src/prepare_rice_kaggle.py`

## 🎓 Use Cases

- **Educational Applications**: Learning tools for rice disease identification
- **Document Processing**: Digitizing agricultural records and field notes
- **Mobile Apps**: Integration into mobile applications for field diagnosis
- **Research**: Basis for more complex agricultural monitoring systems
- **Accessibility**: Assisting farmers with limited access to agricultural experts
- **Precision Agriculture**: Automated crop health monitoring systems

## 🤝 Contributing

This project is open for contributions! Areas for improvement:

- Expanding disease support
- Improving model accuracy
- Adding more training data
- Enhancing UI/UX
- Supporting full plant/symptom recognition
- Adding severity estimation
- Mobile app development
- Model explainability (Grad-CAM visualizations)

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source. Please refer to the license file for details.

## 🙏 Acknowledgments

- **Dataset**: Rice Disease Dataset from Kaggle
- **PyTorch** and **torchvision** communities for the deep learning framework
- **Streamlit** for the web application framework
- **EfficientNet** and **ResNet** authors for pre-trained models
- Contributors and users of this project

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ using PyTorch & Streamlit**
