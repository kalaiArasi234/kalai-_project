# 🔬 Diabetic Retinopathy Detection Project

A deep learning project using ResNet50 to classify diabetic retinopathy severity from retinal fundus images.

## 📋 Project Overview

This project implements a computer vision model to detect and classify diabetic retinopathy into 5 categories:
- **No_DR**: No Diabetic Retinopathy
- **Mild**: Mild Diabetic Retinopathy  
- **Moderate**: Moderate Diabetic Retinopathy
- **Severe**: Severe Diabetic Retinopathy
- **Proliferate_DR**: Proliferative Diabetic Retinopathy

## 🏗️ Project Structure

```
kalai_project/
├── kalai_project.py          # Main training script
├── evaluate_model.py        # Model evaluation with metrics
├── app.py                   # Streamlit web application
├── test_inference.py        # Single image inference testing
├── dr_resnet50.pth         # Trained model weights
├── confusion_matrix.png    # Generated confusion matrix
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- pandas
- streamlit
- seaborn
- matplotlib
- PIL (Pillow)

### Installation

```bash
# Install required packages
pip install torch torchvision scikit-learn pandas streamlit seaborn matplotlib pillow
```

## 📊 Model Performance

The trained ResNet50 model achieves excellent performance on the validation set:

- **Overall Accuracy**: 96.04%

### Per-class Performance:
| Class          | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Mild           | 87.30%    | 94.83% | 90.91%   |
| Moderate       | 95.67%    | 94.31% | 94.99%   |
| No_DR          | 99.18%    | 99.73% | 99.45%   |
| Proliferate_DR | 93.62%    | 86.27% | 89.80%   |
| Severe         | 87.76%    | 87.76% | 87.76%   |

## 🔧 Usage

### 1. Training the Model

```bash
python fazi_project.py
```

This will:
- Load the APTOS 2019 dataset
- Train a ResNet50 model for 5 epochs
- Save the trained model as `dr_resnet50.pth`

### 2. Evaluating the Model

```bash
python evaluate_model.py
```

This will:
- Load the trained model
- Evaluate on the validation set
- Display detailed metrics
- Generate and save a confusion matrix

### 3. Running the Streamlit App

```bash
streamlit run app.py
```

This will:
- Start a web interface at `http://localhost:8501`
- Allow you to upload retinal images
- Display predictions with confidence scores
- Show all class probabilities

### 4. Single Image Inference

```bash
python test_inference.py
```

This demonstrates how to run inference on individual images programmatically.

## 🌐 Streamlit Web Application

The Streamlit app provides a user-friendly interface with:

- **Image Upload**: Support for PNG, JPG, and JPEG formats
- **Real-time Prediction**: Instant classification with confidence scores
- **Probability Visualization**: Bar charts showing all class probabilities
- **Clinical Interpretation**: Helpful descriptions for each severity level
- **Responsive Design**: Works on desktop and mobile devices

### Features:
- 🎯 Color-coded predictions based on severity
- 📊 Interactive probability displays
- 🩺 Clinical interpretation and recommendations
- ⚠️ Medical disclaimer for safety

## 🔬 Technical Details

### Model Architecture
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Modification**: Final layer replaced with 5-class classifier
- **Input Size**: 224×224 pixels
- **Normalization**: ImageNet standard normalization

### Data Preprocessing
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Training Configuration
- **Optimizer**: Adam (lr=1e-4)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 5
- **Batch Size**: 32
- **Train/Val Split**: 80/20
- **Device**: MPS (Apple Silicon) / CUDA / CPU

## 📁 Dataset

The model is trained on the APTOS 2019 Blindness Detection dataset:
- **Total Images**: 3,662 retinal fundus images
- **Image Size**: 224×224 pixels (resized and normalized)
- **Classes**: 5 severity levels of diabetic retinopathy

## 🔍 Key Functions

### `evaluate_model.py`
- `load_trained_model()`: Loads the saved model weights
- `evaluate_model()`: Computes accuracy, precision, recall, F1-score
- `plot_confusion_matrix()`: Visualizes classification results
- `inference_single_image()`: Predicts single image class

### `app.py`
- `load_model()`: Cached model loading for Streamlit
- `preprocess_image()`: Image preprocessing pipeline
- `predict_image()`: End-to-end prediction function

## ⚠️ Important Notes

1. **Medical Disclaimer**: This is a research tool only. Always consult qualified medical professionals for diagnosis and treatment.

2. **Model Security**: The current implementation uses `torch.load` with `weights_only=False`. For production use, consider using `weights_only=True` for enhanced security.

3. **Performance**: The model achieves high accuracy but should not replace professional medical examination.

## 🎯 Future Improvements

- [ ] Implement gradient-based attention visualization
- [ ] Add data augmentation for improved robustness
- [ ] Deploy to cloud platforms (AWS, GCP, Azure)
- [ ] Add batch processing capability
- [ ] Implement model versioning and A/B testing
- [ ] Add more detailed clinical recommendations

## 📄 License

This project is for educational and research purposes. Please ensure proper licensing when using with medical data.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Built with ❤️ using PyTorch and Streamlit**
