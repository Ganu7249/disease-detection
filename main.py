import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import cv2

# Define the paths for the dataset
train_dir = "E:/4_B.Tech Mega Project/PRE-defined/12345/"

# Defining the directories for each disease class
class_directories = {
    "Tomato_healthy": "Tomato_healthy",
    "Target_Spot": "Tomato__Target_Spot",
    "Mosaic_Virus": "Tomato__Tomato_mosaic_virus",
    "YellowLeaf_Curl_Virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Bacterial_Spot": "Tomato_Bacterial_spot",
    "Early_Blight": "Tomato_Early_blight",
    "Late_Blight": "Tomato_Late_blight",
    "Leaf_Mold": "Tomato_Leaf_Mold",
    "Septoria_Leaf_Spot": "Tomato_Septoria_leaf_spot",
    "Spider_Mites": "Tomato_Spider_mites_Two_spotted_spider_mite"
}

# Initialize ImageDataGenerator for augmentation and data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Use 80% for training
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Use 20% for validation
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes (healthy + 9 disease types)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the trained model
model.save("tomato_disease_model.h5")

# Plot training accuracy and loss
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Function to predict the disease from an image
def predict_disease(img_path, model, class_labels):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    
    predictions = model.predict(img_array)
    max_index = np.argmax(predictions)
    confidence = np.max(predictions)

    if confidence > 0.6:  # Set a threshold to handle unrecognized cases
        return class_labels[max_index]
    else:
        return "Disease cannot be recognized due to low confidence"

# Example usage: Testing the model on new images
class_labels = list(class_directories.keys())
model = tf.keras.models.load_model("tomato_disease_model.h5")

# Load a test image (Replace with the path of the test image)
test_image_path = "E:/4_B.Tech Mega Project/your_test_image.jpg"

# Predict disease
predicted_disease = predict_disease(test_image_path, model, class_labels)
print(f"Predicted Disease: {predicted_disease}")

# Show the test image and prediction
img = cv2.imread(test_image_path)
cv2.putText(img, f'Disease: {predicted_disease}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
cv2.imshow("Disease Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
