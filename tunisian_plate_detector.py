import cv2
import easyocr
import numpy as np
import re
import os
import argparse

class TunisianPlateDetector:
    def __init__(self):
        # Initialize the Haar Cascade model for license plate detection
        cascade_path = os.path.join(os.path.dirname(__file__), "model/haarcascade_russian_plate_number.xml")
        self.plate_cascade = cv2.CascadeClassifier(cascade_path)
        # Initialize EasyOCR with Arabic and English support
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False, recog_network='arabic_g1')
        # The Arabic letters in "تونس" (Tunisia)
        self.tunisia_letters = ['ت', 'و', 'ن', 'س']
        
    def detect_plate(self, img):
        """Detect license plate regions in the image using Haar Cascade"""
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(img_gray, 1.1, 4)
        
        plate_regions = []
        for (x, y, w, h) in plates:
            area = w * h
            if area > 500:  # Minimum area threshold
                plate_regions.append((x, y, w, h))
                
        return plate_regions

    def preprocess_image(self, img):
        """Apply multiple preprocessing techniques to improve Arabic detection"""
        # Make sure image is large enough
        if img.shape[1] < 600:
            scale = 600 / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        
        # Create a list of preprocessed variants
        preprocessed_images = []
        
        # 1. Original image
        preprocessed_images.append(img)
        
        # 2. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        sharpened = cv2.filter2D(img, -1, kernel)
        preprocessed_images.append(sharpened)
        
        return preprocessed_images

    def detect_tunisia_text(self, text):
        """Detect if text contains تونس or its letters and replace if needed"""
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
            return "تونس", True
            
        return text, False

    def process_plate_text(self, ocr_results):
        """Process OCR results to extract plate information"""
        # Look for تونس or Arabic letters
        tunisia_text = ""
        tunisia_bbox = None
        was_replaced = False
        
        # First check for exact تونس match
        for bbox, text, prob in ocr_results:
            if 'تونس' in text:
                tunisia_text = text
                tunisia_bbox = bbox
                break
        
        # If no exact match, look for letters
        if not tunisia_text:
            for bbox, text, prob in ocr_results:
                text, replaced = self.detect_tunisia_text(text)
                if replaced:
                    tunisia_text = text
                    tunisia_bbox = bbox
                    was_replaced = True
                    break
        
        # Extract all numbers
        all_text = ' '.join([item[1] for item in ocr_results])
        numbers = re.findall(r'\d+', all_text)
        
        # Format the complete plate text
        if tunisia_text and numbers:
            if len(numbers) >= 2:
                formatted_text = f"{numbers[0]} {tunisia_text} {numbers[1]}"
            else:
                formatted_text = f"{tunisia_text} {' '.join(numbers)}"
        elif numbers:
            formatted_text = f"Plate: {' '.join(numbers)}"
        else:
            formatted_text = tunisia_text if tunisia_text else "Unknown plate"
            
        return formatted_text, tunisia_text, numbers, tunisia_bbox, was_replaced

    def create_plate_visualization(self, plate_text, numbers, tunisia_text):
        """Create a clean visual representation of the license plate"""
        height, width = 180, 500
        plate_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add border
        cv2.rectangle(plate_img, (1, 1), (width-2, height-2), (0, 0, 0), 2)
        
        # Format numbers
        first_num = numbers[0] if numbers else ""
        second_num = numbers[1] if len(numbers) > 1 else ""
        
        # Draw the Tunisia text in the center
        tunisia = tunisia_text if tunisia_text else "تونس"
        cv2.putText(plate_img, tunisia, (width//2-40, height//2+15), 
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

    def save_results(self, img_roi, plate_text, plate_viz, count=0):
        """Save detection results to files"""
        # Create output directory if it doesn't exist
        if not os.path.exists("output"):
            os.makedirs("output")
            
        # Save the plate ROI
        roi_path = os.path.join("output", f"plate_roi_{count}.jpg")
        cv2.imwrite(roi_path, img_roi)
        
        # Save the visualization
        viz_path = os.path.join("output", f"plate_viz_{count}.jpg")
        cv2.imwrite(viz_path, plate_viz)
        
        print(f"Saved detection results:")
        print(f"- ROI: {roi_path}")
        print(f"- Visualization: {viz_path}")
        print(f"- Text: {plate_text}")

    def process_image(self, img_path):
        """Process a single image file"""
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image at {img_path}")
            return None
            
        # Detect plate regions
        plate_regions = self.detect_plate(img)
        if not plate_regions:
            print("No license plates detected")
            return img
            
        result_img = img.copy()
        
        for (x, y, w, h) in plate_regions:
            # Extract and preprocess plate region
            plate_roi = img[y:y+h, x:x+w]
            preprocessed_variants = self.preprocess_image(plate_roi)
            
            # Apply OCR to all variants
            all_results = []
            for variant in preprocessed_variants:
                results = self.reader.readtext(variant)
                if results:
                    all_results.extend(results)
            
            if all_results:
                # Process the text
                plate_text, tunisia_text, numbers, tunisia_bbox, was_replaced = self.process_plate_text(all_results)
                
                # Draw plate rectangle
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw plate text
                text_color = (0, 255, 255) if was_replaced else (255, 0, 255)
                cv2.putText(result_img, plate_text, (x, y-10), 
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, text_color, 2)
                
                # Create clean plate visualization
                plate_viz = self.create_plate_visualization(plate_text, numbers, tunisia_text)
                
                # Show results
                cv2.imshow("License Plate", plate_roi)
                cv2.imshow("Plate Visualization", plate_viz)
                
                return result_img, plate_roi, plate_viz, plate_text
        
        return result_img, None, None, None

    def process_video(self, video_source=0):
        """Process video from camera or file"""
        cap = cv2.VideoCapture(video_source)
        
        cap.set(3, 640)  # width
        cap.set(4, 480)  # height
        
        count = 0
        
        while True:
            success, img = cap.read()
            if not success:
                break
                
            plate_regions = self.detect_plate(img)
            
            if plate_regions:
                for (x, y, w, h) in plate_regions:
                    # Extract and preprocess plate region
                    plate_roi = img[y:y+h, x:x+w]
                    preprocessed_variants = self.preprocess_image(plate_roi)
                    
                    # Apply OCR to all variants
                    all_results = []
                    for variant in preprocessed_variants:
                        results = self.reader.readtext(variant)
                        if results:
                            all_results.extend(results)
                    
                    if all_results:
                        # Process the text
                        plate_text, tunisia_text, numbers, tunisia_bbox, was_replaced = self.process_plate_text(all_results)
                        
                        # Draw rectangle and text
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        text_color = (0, 255, 255) if was_replaced else (255, 0, 255)
                        cv2.putText(img, plate_text, (x, y-10), 
                                  cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, text_color, 2)
                        
                        # Show plate ROI
                        cv2.imshow("License Plate", plate_roi)
                        
                        # Show clean visualization
                        plate_viz = self.create_plate_visualization(plate_text, numbers, tunisia_text)
                        cv2.imshow("Plate Visualization", plate_viz)
            
            cv2.imshow("Tunisian Plate Detector", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and plate_regions:
                self.save_results(plate_roi, plate_text, plate_viz, count)
                count += 1
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Tunisian License Plate Detection')
    parser.add_argument('-i', '--image', help='Path to image file')
    parser.add_argument('-v', '--video', help='Path to video file (omit for webcam)')
    parser.add_argument('-s', '--save', action='store_true', help='Save the detected plates')
    
    args = parser.parse_args()
    detector = TunisianPlateDetector()
    
    if args.image:
        result_img, plate_roi, plate_viz, plate_text = detector.process_image(args.image)
        if result_img is not None:
            cv2.imshow("Result", result_img)
            if args.save and plate_roi is not None:
                detector.save_results(plate_roi, plate_text, plate_viz)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Use webcam or video file
        video_source = 0 if args.video is None else args.video
        detector.process_video(video_source)

if __name__ == "__main__":
    main()
