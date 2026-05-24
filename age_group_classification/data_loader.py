import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

project_dir = Path.cwd()

train_dir = project_dir / "pre-processed/train"
val_dir = project_dir / "pre-processed/valid"
test_dir = project_dir / "pre-processed/test"

# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation/Test preprocessing
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# TRAIN DATA
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'   # MULTI-CLASS
)

# VALIDATION DATA
val_data = test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'   # MULTI-CLASS
)

# TEST DATA
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # MULTI-CLASS
    shuffle=False
)

# Print class mapping
print("Class Indices:")
print(train_data.class_indices)