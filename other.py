import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding to get a binary image (black and white)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return binary

def segment_characters(binary_image):
    # Find contours in the image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by their x-coordinate
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    characters = []
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        char = binary_image[y:y+h, x:x+w]
        
        # Resize the character to a standard size (e.g., 28x28)
        char_resized = cv2.resize(char, (28, 28))
        characters.append(char_resized)
    
    return characters
