from race_classification.data_loader import train_data, val_data
from race_classification.model_gender import build_model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

model = build_model()

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights
)

model.save("race_classification_model.h5")