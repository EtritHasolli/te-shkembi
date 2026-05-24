import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from data_loader import test_data

# Load trained model
model = load_model("race_classification_model.h5")

# True labels
y_true = test_data.classes

# Predict probabilities (softmax output)
y_prob = model.predict(test_data)

# Convert probabilities → class index
y_pred = np.argmax(y_prob, axis=1)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print("\nTEST ACCURACY:", acc)

# Class names (auto from folder structure)
class_names = list(test_data.class_indices.keys())

print("\nCLASSIFICATION REPORT (TEST SET):")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()