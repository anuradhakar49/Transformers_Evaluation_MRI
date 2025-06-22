


### Funtion to estimate IoU for evaluating segmentation:

```python
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

---

```
## To estimate the IOU, you need the ground truth segmentation mask image (y_true) and the predicted segmentation mask image (y_pred)
![image](https://github.com/user-attachments/assets/138ec646-12f1-480b-887d-38754404fcdd)

### Funtion to classfication accuracy and precision

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred)) # y_test are the ground truth and pred are the predicted labels by the classfier

