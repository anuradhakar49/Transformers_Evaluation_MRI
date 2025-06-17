
# 🧠 Brain Tumor Classification with MRI Scans

This repository contains code, datasets, and trained models for classifying brain tumors using MRI images. It includes preprocessing, training, evaluation, and visualization tools built using TensorFlow and Keras.

## 📂 Repository Structure

```
brain-tumor-classification/
│
├── data/
│   ├── train/           # Training images (organized by class)
│   ├── val/             # Validation images
│   └── test/            # Test images
│
├── models/
│   ├── efficientnet_b0_model.h5   # Trained EfficientNetB0 model
│   └── vit_model.h5               # Trained Vision Transformer model
│
├── notebooks/
│   ├── data_preprocessing.ipynb   # Preprocessing and augmentation
│   ├── model_training.ipynb       # Training and validation scripts
│   └── model_evaluation.ipynb     # Testing and performance metrics
│
├── utils/
│   └── helpers.py                 # Utility functions for data loading, plotting, etc.
│
├── requirements.txt
└── README.md
```

## 📊 Dataset

The dataset consists of grayscale MRI scans categorized into multiple brain tumor types. It is split into:
- `train/`: For training the model.
- `val/`: For tuning hyperparameters and validation.
- `test/`: For final evaluation of model performance.

Each folder contains subdirectories corresponding to tumor classes (e.g., `glioma`, `meningioma`, `pituitary`, `no_tumor`).

## 🧠 Models

This repository includes trained models:
- **EfficientNetB0**: Lightweight CNN model using transfer learning.
- **Vision Transformer (ViT)**: Attention-based model adapted for image patch inputs.

Both models were trained using categorical crossentropy loss and optimized using the Adam optimizer.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/brain-tumor-classification.git
   cd brain-tumor-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch notebooks to preprocess data or train/evaluate models.

## 📈 Results

Evaluation metrics include:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

The best performing model achieved over **95% validation accuracy** and strong generalization on the test set.

## 📌 Future Work

- Model explainability with Grad-CAM
- Real-time deployment using a web app (e.g., Streamlit)
- Integration with clinical metadata

## 🧾 License

This project is licensed under the MIT License.

## 🤝 Acknowledgements

- MRI datasets from [Kaggle](https://www.kaggle.com/)
- TensorFlow/Keras team for deep learning libraries
- Research on Vision Transformers and EfficientNet
