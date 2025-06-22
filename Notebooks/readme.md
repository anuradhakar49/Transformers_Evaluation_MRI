
```

---

## 🔍 

```python
Funtion to estimate IoU for evaluating segmentation:

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou


---

```

