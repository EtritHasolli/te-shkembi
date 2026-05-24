import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from data_loader import test_data

# Load trained model
model = load_model("gender_model.h5")

y_true = test_data.classes
y_prob = model.predict(test_data)
y_pred = (y_prob > 0.5).astype(int).flatten()


acc = accuracy_score(y_true, y_pred)
print("\nTEST ACCURACY:", acc)

print("\nCLASSIFICATION REPORT (TEST SET):")
print(classification_report(y_true, y_pred, target_names=["Female", "Male"]))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Female", "Male"],
            yticklabels=["Female", "Male"])

plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
