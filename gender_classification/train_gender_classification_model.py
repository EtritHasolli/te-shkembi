from gender_classification.data_loader import train_data, val_data
from gender_classification.model_gender import build_model

model = build_model()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

model.save("gender_model.h5")