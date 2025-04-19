import cv2
import easyocr
import numpy as np
import re
import os

harcascade = "model/haarcascade_russian_plate_number.xml"

def detect_tunisia_from_letters(text):
    """
    Automatically replaces individual Arabic letters from تونس with the full word
    """
    # Check for any Arabic characters
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    if not arabic_pattern.search(text):
        return "", False
        
    # Individual letters in تونس
    tunisia_letters = ['ت', 'و', 'ن', 'س']
    
    # If text already contains تونس
    if 'تونس' in text:
        return text, False
    
    # Count how many letters from تونس are in the text
    matching_letters = sum(1 for letter in tunisia_letters if letter in text)
    
    # If at least one letter is found, replace with full تونس
    if matching_letters > 0:
        print(f"Found {matching_letters} letters from تونس in '{text}'")
        print(f"Automatically replacing with 'تونس'")
        return "تونس", True
    
    return text, False

def extract_arabic_text(img_roi):
    """Extract Arabic text from license plate region"""
    # Initialize EasyOCR with Arabic as primary language
    reader = easyocr.Reader(['ar', 'en'], gpu=False)
    
    # Preprocess the ROI to improve Arabic detection
    # Resize if too small
    if img_roi.shape[1] < 300:  # if width is less than 300px
        scale = 300 / img_roi.shape[1]
        img_roi = cv2.resize(img_roi, (0, 0), fx=scale, fy=scale)
    
    # Apply some light blurring to reduce noise
    img_processed = cv2.GaussianBlur(img_roi, (3, 3), 0)
    
    # Create multiple preprocessing variants
    img_variants = [
        img_roi,  # Original
        img_processed,  # Blurred
        cv2.cvtColor(cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),  # Grayscale
    ]
    
    all_results = []
    for i, img_var in enumerate(img_variants):
        # Perform OCR
        results = reader.readtext(img_var)
        if results:
            all_results.extend(results)
    
    # Arabic text pattern
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    arabic_text = ""
    arabic_bbox = None
    was_replaced = False
    
    # First, look for exact تونس matches
    for bbox, text, prob in all_results:
        if 'تونس' in text:
            print(f"✓ Found exact match 'تونس' in '{text}' with confidence {prob:.2f}")
            arabic_text = text
            arabic_bbox = bbox
            break
    
    # If no exact match, look for any Arabic text
    if not arabic_text:
        for bbox, text, prob in all_results:
            if arabic_pattern.search(text):
                print(f"Found Arabic text: '{text}' with confidence {prob:.2f}")
                # Check for individual letters and potentially replace
                replaced_text, was_replaced = detect_tunisia_from_letters(text)
                if was_replaced:
                    arabic_text = replaced_text
                    arabic_bbox = bbox
                    break
                else:
                    # Only use if we haven't found better
                    if not arabic_text:
                        arabic_text = text
                        arabic_bbox = bbox
    
    # If still no Arabic found, check if any result has any of the letters
    if not arabic_text:
        for bbox, text, prob in all_results:
            replaced_text, was_replaced = detect_tunisia_from_letters(text)
            if was_replaced:
                arabic_text = replaced_text
                arabic_bbox = bbox
                break
    
    # If still nothing found but we have results, force "تونس" on the first detection
    if not arabic_text and all_results:
        arabic_text = "تونس"  # Force "Tunisia" text
        arabic_bbox = all_results[0][0]  # Use first detection's bbox
        was_replaced = True
        print("No Arabic detected, forcing 'تونس' for Tunisian plate")
    
    # Extract numbers using regex from all detections
    all_text = ' '.join([item[1] for item in all_results])
    numbers = re.findall(r'\d+', all_text)
    
    # Prepare formatted plate text
    if arabic_text and numbers:
        plate_text = f"{arabic_text} {' '.join(numbers)}"
    elif arabic_text:
        plate_text = arabic_text
    elif numbers:
        plate_text = f"Plate: {' '.join(numbers)}"
    else:
        plate_text = "Unknown"
    
    # Create a result image with annotations
    result_img = img_roi.copy()
    
    # Draw all detected text
    for (bbox, text, prob) in all_results:
        points = np.array(bbox, dtype=np.int32)
        cv2.polylines(result_img, [points], True, (0, 255, 255), 2)
        
        # Position for text label
        x_min, y_min = points.min(axis=0)
        cv2.putText(result_img, f"{text} ({prob:.2f})", 
                   (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # Highlight Arabic text if found
    if arabic_bbox:
        points = np.array(arabic_bbox, dtype=np.int32)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Green box for Arabic (thicker if replaced)
        thickness = 3 if was_replaced else 2
        cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness)
        
        # Add a label for the Arabic text
        label = f"{arabic_text}"
        if was_replaced:
            label += " (auto)"
        cv2.putText(result_img, label, (x_min, y_min-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Extract just the Arabic portion
        margin = 5  # Add a small margin
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(result_img.shape[1], x_max + margin)
        y_max = min(result_img.shape[0], y_max + margin)
        arabic_region = img_roi[y_min:y_max, x_min:x_max]
    else:
        arabic_region = None
    
    return result_img, plate_text, arabic_text, numbers, arabic_region, was_replaced

def create_plate_output(plate_text, numbers, arabic_text):
    """Create a clean visual representation of the license plate"""
    # Create a blank image with license plate dimensions
    height, width = 180, 500
    plate_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Add a border
    cv2.rectangle(plate_img, (1, 1), (width-2, height-2), (0, 0, 0), 2)
    
    # Format for Tunisian plate: First number + تونس + second number
    first_num = ""
    second_num = ""
    
    if len(numbers) >= 1:
        first_num = numbers[0]
    if len(numbers) >= 2:
        second_num = numbers[1]
    
    # Draw the Tunisia text in the center
    tunisia_text = arabic_text if arabic_text else "تونس"
    cv2.putText(plate_img, tunisia_text, (width//2-40, height//2+15), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 150), 2)
    
    # Draw numbers
    if first_num:
        cv2.putText(plate_img, first_num, (width//4-30, height//2+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    if second_num:
        cv2.putText(plate_img, second_num, (3*width//4-30, height//2+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Add "TUNISIA" text at the bottom
    cv2.putText(plate_img, "TUNISIA", (width//2-50, height-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return plate_img

def main():
    cap = cv2.VideoCapture(0)

    cap.set(3, 640) # width
    cap.set(4, 480) # height

    min_area = 500
    count = 0
    
    # Create output directory if it doesn't exist
    if not os.path.exists("plates"):
        os.makedirs("plates")
    
    if not os.path.exists("arabic_regions"):
        os.makedirs("arabic_regions")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
        
        for (x,y,w,h) in plates:
            area = w * h

            if area > min_area:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                
                # Extract plate region
                img_roi = img[y: y+h, x:x+w]
                
                # Process the ROI to detect Arabic text
                result_roi, plate_text, arabic_text, numbers, arabic_region, was_replaced = extract_arabic_text(img_roi)
                
                # Show the processed ROI with text detections
                cv2.imshow("ROI", result_roi)
                
                # Generate and display the plate output
                plate_output = create_plate_output(plate_text, numbers, arabic_text)
                cv2.imshow("Plate Output", plate_output)
                
                # Display plate text on main image
                text_color = (255, 0, 255)  # Default
                if was_replaced:  # Use a different color for auto-replaced text
                    text_color = (0, 255, 255)  # Yellow for auto-replaced
                
                cv2.putText(img, plate_text, (x, y-5), 
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, text_color, 2)
                
                # If Arabic text found, show it separately
                if arabic_region is not None:
                    cv2.imshow("Arabic", arabic_region)

        cv2.imshow("Result", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the plate image
            plate_path = f"plates/scaned_img_{count}.jpg"
            cv2.imwrite(plate_path, img_roi)
            
            # Save the Arabic region if available
            if arabic_region is not None:
                arabic_path = f"arabic_regions/arabic_{count}.jpg"
                cv2.imwrite(arabic_path, arabic_region)
            
            # Save the clean plate output
            plate_output_path = f"plates/plate_output_{count}.jpg"
            cv2.imwrite(plate_output_path, plate_output)
            
            cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
            cv2.putText(img, "Plate Saved", (150, 265), 
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Results", img)
            cv2.waitKey(500)
            count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

