
# 🤗 Accessing Trained Models from Hugging Face Hub or uploaded here

This guide shows how to load and use pre-trained models from the [Hugging Face Model Hub](https://huggingface.co/models) using the `transformers` library. Some trained models are directly uploaded here which can be used for inference.

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

## 📚 References

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)

