import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ✅ Set paths
base_dir = "C:\\Users\\Admin\\Desktop\\internship_projects\\cats dogs classifeir\\cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# ✅ Image data generators with rescaling
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# ✅ Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# ✅ Define CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# ✅ Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# ✅ Save the model in a custom folder
model_dir = "C:\\Users\\Admin\\Desktop\\internship_projects\\cats dogs classifeir\\saved_models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "cat_dog_classifier_model.h5")
model.save(model_path)

print(f"✅ Model saved to: {model_path}")
