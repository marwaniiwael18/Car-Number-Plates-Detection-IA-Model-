import cv2
import easyocr
import numpy as np
import re
import argparse

class ArabicTextExtractor:
    """
    A class to extract Arabic text from Tunisian license plates
    """
    
    def __init__(self):
        # Initialize EasyOCR with Arabic as primary language
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False)
        
    def preprocess_image(self, img):
        """Apply preprocessing to improve Arabic text detection"""
        # Resize if too small
        if img.shape[1] < 600:  # if width is less than 600px
            scale = 600 / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        
        # Apply some light blurring to reduce noise
        img_processed = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Enhance contrast
        lab = cv2.cvtColor(img_processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_img
    
    def extract_arabic_text(self, ocr_result):
        """Extract Arabic text from OCR results"""
        arabic_text = ""
        highest_conf = 0
        arabic_bbox = None
        
        # Arabic text pattern
        arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
        
        for bbox, text, prob in ocr_result:
            # Check if text contains Arabic characters
            if arabic_pattern.search(text) or "تونس" in text:
                print(f"Found Arabic text: '{text}' with confidence {prob:.2f}")
                # Keep track of the highest confidence Arabic text
                if prob > highest_conf:
                    highest_conf = prob
                    arabic_text = text
                    arabic_bbox = bbox
        
        # If no Arabic found, try to find تونس in all text
        if not arabic_text:
            all_text = ' '.join([item[1] for item in ocr_result])
            tunisia_match = re.search(r'تونس', all_text)
            if tunisia_match:
                arabic_text = "تونس"
        
        return arabic_text, arabic_bbox
    
    def process_plate(self, ocr_result):
        """Process the complete license plate"""
        # Extract the Arabic text
        arabic_text, arabic_bbox = self.extract_arabic_text(ocr_result)
        
        # Combine all detected text
        all_text = ' '.join([item[1] for item in ocr_result])
        
        # Extract numbers using regex
        numbers = re.findall(r'\d+', all_text)
        
        if arabic_text and numbers:
            # Format as "تونس number1 number2"
            formatted_plate = f"{arabic_text} {' '.join(numbers)}"
            return formatted_plate, arabic_text, numbers, arabic_bbox
        elif numbers:
            # If Arabic text not found but we have numbers
            return f"Plate numbers: {' '.join(numbers)}", "", numbers, None
        else:
            return "No valid Tunisian plate detected", "", [], None
    
    def extract_from_image(self, image_path):
        """Extract Arabic text from a license plate image"""
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None, "", "", []
        
        # Preprocess the image
        preprocessed_img = self.preprocess_image(img)
        
        # Perform OCR
        results = self.reader.readtext(preprocessed_img)
        
        if not results:
            # Try again with the original image if no results
            results = self.reader.readtext(img)
        
        # Process the plate
        plate_text, arabic_text, numbers, arabic_bbox = self.process_plate(results)
        
        # Create a visualization
        img_result = img.copy()
        
        # Draw all detected text areas
        for bbox, text, _ in results:
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(img_result, [points], True, (0, 255, 255), 2)
            
            # Position for text label
            x_min, y_min = points.min(axis=0)
            cv2.putText(img_result, text, 
                       (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Highlight Arabic text if found
        if arabic_bbox:
            points = np.array(arabic_bbox, dtype=np.int32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            # Green box for Arabic
            cv2.rectangle(img_result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            cv2.putText(img_result, f"Arabic: {arabic_text}", 
                       (x_min, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Extract just the Arabic portion
            arabic_region = img[y_min:y_max, x_min:x_max]
        else:
            arabic_region = None
        
        # Display the full processed plate text at the top
        cv2.putText(img_result, f"Plate: {plate_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display numbers separately
        if numbers:
            cv2.putText(img_result, f"Numbers: {' '.join(numbers)}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return img_result, arabic_region, plate_text, arabic_text, numbers
    
    def extract_from_video(self, video_source=0):
        """Extract Arabic text from video stream"""
        cap = cv2.VideoCapture(video_source)
        
        cap.set(3, 640)  # width
        cap.set(4, 480)  # height
        
        while True:
            success, img = cap.read()
            if not success:
                break
            
            # Preprocess
            preprocessed_img = self.preprocess_image(img)
            
            # Perform OCR
            results = self.reader.readtext(preprocessed_img)
            
            # Process results
            if results:
                plate_text, arabic_text, numbers, arabic_bbox = self.process_plate(results)
                
                # Highlight Arabic text if found
                if arabic_bbox:
                    points = np.array(arabic_bbox, dtype=np.int32)
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)
                    
                    # Green box for Arabic
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    cv2.putText(img, f"Arabic: {arabic_text}", 
                              (x_min, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Show the Arabic region
                    arabic_region = img[y_min:y_max, x_min:x_max]
                    cv2.imshow("Arabic Text", arabic_region)
                
                # Show plate info
                cv2.putText(img, f"Plate: {plate_text}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow("Arabic Text Extractor", img)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Extract Arabic Text from Tunisian License Plates')
    parser.add_argument('-i', '--image', help='Path to image file')
    parser.add_argument('-v', '--video', help='Path to video file (omit for webcam)')
    parser.add_argument('-s', '--save', action='store_true', help='Save the Arabic text region')
    
    args = parser.parse_args()
    
    extractor = ArabicTextExtractor()
    
    if args.image:
        result_img, arabic_region, plate_text, arabic_text, numbers = extractor.extract_from_image(args.image)
        
        print(f"Detected plate: {plate_text}")
        print(f"Arabic text: {arabic_text}")
        print(f"Numbers: {numbers}")
        
        cv2.imshow("Result", result_img)
        
        if arabic_region is not None:
            cv2.imshow("Arabic Region", arabic_region)
            
            if args.save:
                output_path = "arabic_text.jpg"
                cv2.imwrite(output_path, arabic_region)
                print(f"Arabic text region saved to {output_path}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Use webcam or video file
        video_source = 0 if args.video is None else args.video
        extractor.extract_from_video(video_source)

if __name__ == "__main__":
    main()
