import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Preprocess the image (grayscale, blur, binarize)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

# Segment characters from the preprocessed image
def segment_characters(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char = binary_image[y:y+h, x:x+w]
        char_resized = cv2.resize(char, (28, 28))
        characters.append(char_resized)
    return characters

# Build the CRNN model
def build_crnn(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Reshape((-1, 128)))  # Flatten
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=False)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(train_images, train_labels, val_images, val_labels, num_classes):
    input_shape = (28, 28, 1)
    model = build_crnn(input_shape, num_classes)

    # Data augmentation for robust training
    datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, fill_mode='nearest'
    )
    
    # Fit the model
    model.fit(datagen.flow(train_images, train_labels, batch_size=32), 
              epochs=50, validation_data=(val_images, val_labels))
    return model

# Example function to break a CAPTCHA using the trained model
def break_captcha(model, captcha_image_path):
    binary_image = preprocess_image(captcha_image_path)
    characters = segment_characters(binary_image)
    predicted_text = ''
    
    for char in characters:
        char = char.reshape(1, 28, 28, 1)
        prediction = model.predict(char)
        predicted_char = chr(prediction.argmax(axis=1)[0] + ord('A'))  # Modify as needed for your dataset
        predicted_text += predicted_char
    
    print(f"Predicted CAPTCHA: {predicted_text}")
    return predicted_text

if __name__ == '__main__':
    # Load your dataset here (you can modify this part based on your dataset structure)
    # train_images, train_labels should be prepared from your dataset
    
    # Example: Train the model (you'll need to prepare train_images, train_labels)
    num_classes = 36  # Assuming alphanumeric CAPTCHAs (10 digits + 26 letters)
    model = train_model(train_images, train_labels, val_images, val_labels, num_classes)
    
    # Example: Break a CAPTCHA using the trained model
    # captcha_image_path = 'path_to_test_captcha_image.jpg'
    # break_captcha(model, captcha_image_path)
