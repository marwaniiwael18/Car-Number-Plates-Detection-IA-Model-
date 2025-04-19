import matplotlib.pyplot as plt
import cv2
import easyocr
from IPython.display import Image
import numpy as np
import re


def preprocess_image(image_path):
    """Apply multiple preprocessing techniques to improve Arabic detection"""
    # Read image
    original = cv2.imread(image_path)
    if original is None:
        return None, None
    
    # Make sure image is large enough
    if original.shape[1] < 600:
        scale = 600 / original.shape[1]
        original = cv2.resize(original, (0, 0), fx=scale, fy=scale)
    
    # Create a list of preprocessed variants
    preprocessed_images = []
    
    # 1. Original image
    preprocessed_images.append(original)
    
    # 2. Grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    preprocessed_images.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    
    # 3. Bilateral filter (reduces noise while keeping edges sharp)
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    preprocessed_images.append(cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR))
    
    # 4. Adaptive threshold (helps with contrast)
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    preprocessed_images.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
    
    # 5. Edge enhancement
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(original, -1, kernel)
    preprocessed_images.append(sharpened)
    
    # Display the preprocessing results
    plt.figure(figsize=(20, 10))
    titles = ['Original', 'Grayscale', 'Bilateral Filter', 'Adaptive Threshold', 'Sharpened']
    
    # for i, img in enumerate(preprocessed_images):
    #     plt.subplot(1, 5, i+1)
    #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     plt.title(titles[i])
    #     plt.axis('off')
    
    # plt.tight_layout()
    # plt.show()
    
    return preprocessed_images, original

# Preprocess the image
preprocessed_images, original_img = preprocess_image('/Users/macbook/Documents/Personnel/Car-Number-Plates-Detection/plates/tunisNT.jpeg')


# Initialize the reader with Arabic as the primary language, followed by English
# Setting decoder parameters to optimize for Arabic text detection
reader = easyocr.Reader(['ar', 'en'], gpu=False, recog_network='arabic_g1')

print("OCR loaded with Arabic support. Processing images...")


# Apply OCR to each preprocessed image and collect all results
all_results = []

for i, img in enumerate(preprocessed_images):
    print(f"Processing image variant {i+1}...")
    # Use detail=1 to get bounding box info
    # Allow lower confidence to catch more potential matches
    results = reader.readtext(img, paragraph=False, detail=1, 
                               min_size=10, contrast_ths=0.1,
                               adjust_contrast=0.5)
    all_results.append(results)
    
    # print(f"Found {len(results)} text regions in variant {i+1}:")
    # for bbox, text, prob in results:
    #     print(f"  Text: '{text}' with confidence {prob:.2f}")

# Get the original OCR results
output = all_results[0]


def extract_numbers_and_insert_tunisia(txt):
    """Extract numbers from OCR results and insert تونس between them"""
    # Collect all detected text
    all_text = txt
    print(f"All detected text: {all_text}")
    
    # Extract numbers using regex
    numbers = re.findall(r'\d+', all_text)
    print(f"Extracted numbers: {numbers}")
    
    # Sort numbers by their occurrence in the text
    # This helps maintain the original order
    sorted_numbers = sorted(numbers, key=lambda x: all_text.find(x))
    
    # Remove duplicates while preserving order
    unique_numbers = []
    for num in sorted_numbers:
        if num not in unique_numbers:
            unique_numbers.append(num)
    
    print(f"Unique numbers in order: {unique_numbers}")
    
    # Build the license plate text with تونس in the middle
    if len(unique_numbers) >= 2:
        # Typical format: first_number تونس second_number
        first_part = unique_numbers[0]
        second_part = unique_numbers[1]
        complete_plate = f"{first_part} تونس {second_part}"
        
        # If there are more numbers, add them too
        if len(unique_numbers) > 2:
            complete_plate += " " + " ".join(unique_numbers[2:])
    elif len(unique_numbers) == 1:
        # Only one number detected
        complete_plate = f"تونس {unique_numbers[0]}"
    else:
        # No numbers detected
        complete_plate = "تونس"
    
    return complete_plate, unique_numbers

# Extract numbers and insert تونس
# plate_text, extracted_numbers = extract_numbers_and_insert_tunisia(all_results[0])
# print(f"\nFormatted plate: {plate_text}")


for bbox, text, prob in results:
    if 'ت' in text and 'و' in text:
        print(f"✓ Found exact match 'تونس' in '{text}' with confidence {prob:.2f}")
        plate_text, extracted_numbers = extract_numbers_and_insert_tunisia(text)
        print(f"\nFormatted plate: {plate_text}")
        break
    elif 'ت' in text and 'ن' in text:
        print(f"Found Arabic text: '{text}' with confidence {prob:.2f}")
        break
    else:
        print(f"Found normal text: '{text}' with confidence {prob:.2f}")
        break
    