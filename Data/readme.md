
# 📊 Accessing MRI classification and segmentation datasets from Hugging Face 🤗

This repository demonstrates how to access and explore datasets hosted on [Hugging Face Hub](https://huggingface.co/datasets) using the `datasets` library in Python.

## 🛠️ Requirements

Install the Hugging Face `datasets` library:

```bash
!pip install -U datasets
```

## 📥 Loading a Dataset

You can load any dataset from the Hugging Face Hub using the `load_dataset()` function. To access a dataset, you will need to create a Huggingface account and generate an access token.

https://huggingface.co/docs/hub/en/security-tokens 

### Example 1: Load the MRI Classification dataset

```python
from datasets import load_dataset
from huggingface_hub import login

login(token='Your access token') 

# Load the MRI Classification dataset
dataset = load_dataset("akar49/MRI_3TumorClassification")
print(dataset)

# View the first training example
plt.imshow(dataset["train"][0])

```

### Example 2: Load the MRI Segmentation dataset

```python
from datasets import load_dataset

# Replace 'username/dataset_name' with the actual dataset path
dataset = load_dataset("akar49/MRI_Segmentation-1")

# Preview the dataset
print(dataset)
```


## 📚 References

- [🤗 Datasets Documentation](https://huggingface.co/docs/datasets)
- [Hugging Face Hub - Datasets](https://huggingface.co/datasets)

---
