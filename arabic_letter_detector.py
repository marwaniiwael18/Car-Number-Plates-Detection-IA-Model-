import cv2
import easyocr
import re
import numpy as np
import argparse

class TunisianArabicDetector:
    """Utility to detect Tunisian license plates with automatic letter replacement"""
    
    def __init__(self):
        """Initialize the detector with EasyOCR"""
        # Load EasyOCR with Arabic as the primary language
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False, recog_network='arabic_g1')
        
        # The Arabic letters in "تونس" (Tunisia)
        self.tunisia_letters = ['ت', 'و', 'ن', 'س']
        
    def detect_letters(self, text):
        """
        Detect any letters from تونس in the given text
        Returns (text, was_replaced) tuple - text will be "تونس" if letters are found
        """
        # Check if the text already contains تونس
        if 'تونس' in text:
            return text, False
            
        # Check for any Arabic text
        arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
        if not arabic_pattern.search(text):
            return text, False
            
        # Look for individual letters from تونس
        matches = []
        for letter in self.tunisia_letters:
            if letter in text:
                matches.append(letter)
                
        # If we found any letters, replace with full word
        if matches:
            print(f"Found letters {', '.join(matches)} from تونس in '{text}'")
            print("Automatically replacing with 'تونس'")
            return "تونس", True
            
        return text, False
        
    def process_image(self, image_path):
        """Process an image to detect Arabic letters and replace with تونس"""
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        # Preprocess image variants
        img_variants = [
            img,  # Original
            cv2.GaussianBlur(img, (3, 3), 0),  # Slight blur
            cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # Grayscale
        ]
        
        # Apply OCR to all image variants
        all_detections = []
        for i, img_var in enumerate(img_variants):
            detections = self.reader.readtext(img_var)
            if detections:
                all_detections.extend(detections)
                print(f"Variant {i+1}: Found {len(detections)} text regions")
                
        # Process results looking for Arabic letters
        result_img = img.copy()
        found_tunisia = False
        tunisia_bbox = None
        
        # First, check for exact تونس
        for bbox, text, prob in all_detections:
            if 'تونس' in text:
                found_tunisia = True
                tunisia_bbox = bbox
                print(f"✓ Found exact match 'تونس' in '{text}'")
                break
                
        # If not found, check for individual letters
        if not found_tunisia:
            for bbox, text, prob in all_detections:
                text, was_replaced = self.detect_letters(text)
                if was_replaced:
                    found_tunisia = True
                    tunisia_bbox = bbox
                    break
        
        # Draw bounding boxes for all detections
        for bbox, text, prob in all_detections:
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(result_img, [points], True, (0, 255, 255), 2)
            
            # Add text label
            x_min, y_min = points.min(axis=0)
            cv2.putText(result_img, f"{text} ({prob:.2f})", 
                       (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Highlight the Tunisia detection if found
        if found_tunisia and tunisia_bbox:
            points = np.array(tunisia_bbox, dtype=np.int32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            # Green box for Tunisia
            cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            cv2.putText(result_img, "TUNISIA: تونس", (x_min, y_min-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                       
            # Extract the Tunisia region
            margin = 5
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img.shape[1], x_max + margin)
            y_max = min(img.shape[0], y_max + margin)
            tunisia_region = img[y_min:y_max, x_min:x_max]
            
            return result_img, tunisia_region
        else:
            cv2.putText(result_img, "No 'تونس' detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return result_img, None
    
def main():
    parser = argparse.ArgumentParser(description="Detect Arabic letters ت and و and replace with تونس")
    parser.add_argument("-i", "--image", required=True, help="Path to input image")
    parser.add_argument("-s", "--save", action="store_true", help="Save the output image")
    
    args = parser.parse_args()
    
    # Initialize the detector
    detector = TunisianArabicDetector()
    
    # Process the image
    result_img, tunisia_region = detector.process_image(args.image)
    
    if result_img is not None:
        # Display the result
        cv2.imshow("Result", result_img)
        
        # Display Tunisia region if found
        if tunisia_region is not None:
            cv2.imshow("Tunisia Region", tunisia_region)
            print("✓ Successfully detected تونس")
            
            # Save the Tunisia region if requested
            if args.save:
                output_path = "tunisia_region.jpg"
                cv2.imwrite(output_path, tunisia_region)
                print(f"✓ Saved Tunisia region to {output_path}")
        
        # Save the complete result if requested
        if args.save:
            output_path = "result_image.jpg"
            cv2.imwrite(output_path, result_img)
            print(f"✓ Saved result image to {output_path}")
            
        # Wait for key press
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
