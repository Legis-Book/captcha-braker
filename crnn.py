import tensorflow as tf
from tensorflow.keras import layers, models

def build_crnn(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers for feature extraction
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    # Reshape for RNN layers
    model.add(layers.Reshape((-1, 128)))  # Flatten along the height and width dimensions

    # Recurrent layers (LSTM)
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=False)))

    # Fully connected layers and output
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Assuming `num_classes` covers all possible CAPTCHA characters

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
