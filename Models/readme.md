
# 🤗 Accessing Trained Models from Hugging Face Hub or uploaded here

This guide shows how to load apre-trained models from the [Hugging Face Model Hub](https://huggingface.co/models) using the `transformers` library. Some trained models are stored on an external drive whose links are provided

## 🛠️ Requirements

Install the required libraries:

```bash
pip install transformers
```

Optional: for PyTorch or TensorFlow support

```bash
pip install torch         # For PyTorch
# OR
pip install tensorflow    # For TensorFlow
```

---

## 🔍 Example 1: Load a trained Segformer model for tumor segmentation

```python

# Load model from Hugging Face Hub
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b0")

model = SegformerForSemanticSegmentation.from_pretrained("akar49/Segformer-pytorch_meningioma_Jun25")
OR
model = SegformerForSemanticSegmentation.from_pretrained("akar49/Segformer-pytorch_pituitary_Jun25")
OR
model = SegformerForSemanticSegmentation.from_pretrained("akar49/Segformer-pytorch_glioma_Jun25")

```

---

## 🔍 Example 2: Load a trained Maskformer model for Glioma segmentation

```python
from transformers import MaskFormerForInstanceSegmentatio

model = MaskFormerForInstanceSegmentation.from_pretrained("akar49/Maskformer-MRI_meningiomaJun25")
OR
model = MaskFormerForInstanceSegmentation.from_pretrained("akar49/Maskformer-MRI_gliomaJun25")
OR
model = MaskFormerForInstanceSegmentation.from_pretrained("akar49/Maskformer-MRI_pituitaryJun25")

---

```

---

## 🔍 Example 3: Load a trained UNet model for tumor segmentation

The trained model may be found in the link: https://drive.google.com/file/d/1NCmgAdcuECyBDD1C3I4oHHm9kiholSzc/view?usp=drive_link

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model

model = load_model('path to/Unet_model.h5')

---

```

---

## 🔍 Example 4: Load a trained CNN  model for tumor classification 
The trained model may be found in the link: https://drive.google.com/file/d/1hOvhnw6BUwYXkcrYPowejcAylQ7K9ICZ/view?usp=drive_link 

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model

model = load_model('path to/cnn_classification_model.h5')

## 📚 References

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)

