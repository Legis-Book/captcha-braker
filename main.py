import cv2
import pytesseract
from pytesseract import Output

# Set path to tesseract if necessary
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # For macOS
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows

# Function to preprocess the image
def preprocess_image(image_path):
    # Read image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to grayscale for better OCR performance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to make the image black and white
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    return thresh

# Function to extract numbers using Tesseract OCR
def extract_numbers_from_image(image_path):
    # Preprocess the image (optional, improves OCR performance)
    processed_image = preprocess_image(image_path)

    # Use Tesseract to do OCR on the processed image
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    ocr_result = pytesseract.image_to_string(processed_image, config=custom_config)

    # Filter out only the numeric characters from the OCR result
    numbers = ''.join(filter(str.isdigit, ocr_result))
    
    return numbers

# Example usage
image_path = 'path_to_image_with_numbers.jpg'
numbers = extract_numbers_from_image(image_path)
print(f"Extracted Numbers: {numbers}")
