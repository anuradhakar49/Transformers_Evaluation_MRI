
# 🧠 Brain Tumor Classification with MRI Scans

This repository contains code, datasets, and trained models for classifying brain tumors using MRI images. It includes preprocessing, training, evaluation, and visualization tools built using TensorFlow and Keras.

## 📂 Repository Structure

```
brain-tumor-classification/
│
├── data/
│   ├── Classification           # Images with labels (organized by class) for brain tumor classification
│   ├── Segmentation             # Images with masks (organized by class) for brain tumor segmentation


├── models/
│   ├── efficientnet_b0_model.h5   # Trained EfficientNetB0 model
│   └── UNet_model.h5               # Trained Vision Transformer model
    └── Vision transformer
    └── Segformer (3 models for three types of tumors: Glioma, Meningioma, Pituitary)
    └── Maskformer (3 models for three types of tumors:Glioma, Meningioma, Pituitary)
│
├── notebooks/
│   ├── data_loading.ipynb   # Preprocessing and augmentation
│   ├── model_inference.ipynb       # Training and validation scripts
│   └── model_evaluation.ipynb     # Testing and performance metrics.
│
├── requirements.txt
└── README.md
```

## 📊 Dataset

The dataset consists of grayscale MRI scans categorized into multiple brain tumor types. The datasets are uploaded under two categories
- MRI Classification: For training the model.
- MRI Segmentation: For tuning hyperparameters and validation.
- Each dataset is split into train, test and validation sets.

The scripts to access the datasets are provided here.

## 🧠 Models

This repository includes trained models:
- **EfficientNetB0**: Lightweight CNN model using transfer learning.
- **Vision Transformer (ViT)**: Attention-based model adapted for image patch inputs.
- **Segformer**: Attention-based model adapted for image segmentation.
- **Maskformer**: Attention-based model adapted for image segmentation.

## 🚀 How to use the resources

1. Create a new python environment:
   ```
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Launch notebooks to preprocess data or train/evaluate models.

## 📈 Results

Evaluation metrics include:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- IoU
These metrics are implement in the notebooks folder

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
