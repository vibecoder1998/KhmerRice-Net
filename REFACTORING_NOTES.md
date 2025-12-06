# Refactoring Notes: Following Khmer Character Classification Structure

## Overview

The project has been refactored to follow the structure from the [Khmer Character Classification project](https://github.com/vibecoder1998/khmer-character-classification).

## New Structure

### Source Code (`src/`)

Following the reference project structure:

- **`model_def.py`** (renamed from `rice_model.py`)
  - Contains `RiceCNN` class definition
  - Neural network architecture for rice disease classification
  
- **`preprocess.py`** (new)
  - Image preprocessing utilities
  - Functions: `preprocess_image()`, `get_transform()`
  - Handles image normalization and transformation
  
- **`inference.py`** (new)
  - Model loading and prediction logic
  - Functions: `load_model_and_metadata()`, `predict()`
  - Handles model checkpoint loading and inference
  
- **`rice_dataset.py`** (kept)
  - PyTorch Dataset class for training
  - Handles data loading and class mapping
  
- **`train_rice.py`** (updated)
  - Training script (updated to use new module names)
  - Can be moved to root or `scripts/` if desired
  
- **`prepare_rice_kaggle.py`** (kept)
  - Dataset preparation script

### Application

- **`app.py`** (moved from `src/app_rice_streamlit.py`)
  - Main Streamlit application
  - Now at root level, following reference structure
  - Uses new `inference` and `preprocess` modules

### Utilities (`utils/`)

- **`label_map.py`** (new, extends `mapping.py`)
  - Disease information with Khmer/English names
  - Advice for each disease
  - Label mappings
  
- **`mapping.py`** (kept for backward compatibility)
  - Original label mappings

## Migration Guide

### For Existing Code

1. **Import Changes:**
   ```python
   # Old
   from src.rice_model import RiceCNN
   
   # New
   from src.model_def import RiceCNN
   ```

2. **Preprocessing:**
   ```python
   # Old (inline transforms)
   transform = transforms.Compose([...])
   
   # New
   from src.preprocess import preprocess_image
   tensor = preprocess_image(image)
   ```

3. **Inference:**
   ```python
   # Old (inline in app)
   model, classes, transform = load_model_and_metadata()
   preds = predict(image, model, transform, classes)
   
   # New
   from src.inference import load_model_and_metadata, predict
   model, classes, transform_info = load_model_and_metadata()
   top_label, confidence, all_probs = predict(model, image, classes)
   ```

4. **Running the App:**
   ```bash
   # Old
   streamlit run src/app_rice_streamlit.py
   
   # New
   streamlit run app.py
   ```

## File Mapping

| Old File | New File | Status |
|----------|----------|--------|
| `src/rice_model.py` | `src/model_def.py` | Renamed |
| `src/app_rice_streamlit.py` | `app.py` | Moved to root |
| - | `src/preprocess.py` | New |
| - | `src/inference.py` | New |
| `utils/mapping.py` | `utils/label_map.py` | Extended |
| `src/train_rice.py` | `src/train_rice.py` | Updated imports |

## Benefits

1. **Clear Separation of Concerns:**
   - Model definition separate from preprocessing
   - Inference logic isolated from application
   - Easier to test and maintain

2. **Reusability:**
   - Preprocessing functions can be used in training and inference
   - Inference logic can be used in different applications
   - Model definition is standalone

3. **Consistency:**
   - Follows established project structure
   - Easier for developers familiar with similar projects
   - Better code organization

## Backward Compatibility

- Old `rice_model.py` can be kept as an alias if needed
- Training script updated to use new imports
- Model checkpoints remain compatible

## Next Steps

1. Test the refactored code
2. Update any documentation referencing old file names
3. Consider moving `train_rice.py` to `scripts/` directory
4. Remove deprecated files after verification

---

**Refactoring Date**: 2024  
**Reference**: [Khmer Character Classification](https://github.com/vibecoder1998/khmer-character-classification)

