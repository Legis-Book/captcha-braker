from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation pipeline
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Assuming `train_images` is your dataset of CAPTCHA images
augmented_images = datagen.flow(train_images, batch_size=32)
