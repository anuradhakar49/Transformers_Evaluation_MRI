
# 🤗 Accessing Trained Models from Hugging Face Hub

This guide shows how to load and use pre-trained models from the [Hugging Face Model Hub](https://huggingface.co/models) using the `transformers` library.

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

## 🔍 Example 1: Load a trained vision transformer model for tumor classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model from Hugging Face Hub
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize and run inference
inputs = tokenizer("This is amazing!", return_tensors="pt")
outputs = model(**inputs)

print(outputs.logits)
```

---

## 🔍 Example 2: Load a trained  Segformer model for Glioma segmentation

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model from Hugging Face Hub
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize and run inference
inputs = tokenizer("This is amazing!", return_tensors="pt")
outputs = model(**inputs)

print(outputs.logits)
```

---

---

## 🔍 Checkpoints for all odels


---

## 📚 References

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)

---

Feel free to contribute by adding more examples or tasks (e.g. question answering, translation, summarization)!

