import cv2
import easyocr
import numpy as np
import re
import argparse

class TunisianPlateDetector:
    def __init__(self):
        # Initialize the Haar Cascade model for license plate detection
        self.plate_cascade = cv2.CascadeClassifier("model/haarcascade_russian_plate_number.xml")
        # Initialize EasyOCR with Arabic and English support
        self.reader = easyocr.Reader(['ar', 'en'])
        
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
    
    def recognize_text(self, img, regions):
        """Extract text from detected plate regions"""
        results = []
        for (x, y, w, h) in regions:
            # Crop the license plate region
            plate_img = img[y:y+h, x:x+w]
            
            # Use EasyOCR to read text
            ocr_result = self.reader.readtext(plate_img)
            
            # Process the OCR result
            plate_text = self.process_tunisian_plate(ocr_result)
            results.append((plate_text, (x, y, w, h), ocr_result))
            
        return results
    
    def process_tunisian_plate(self, ocr_result):
        """Format detected text as Tunisian license plate"""
        if not ocr_result:
            return "No text detected"
        
        # Combine all detected text
        all_text = ' '.join([item[1] for item in ocr_result])
        
        # Look for تونس (Tunisia in Arabic)
        tunisia_pattern = re.search(r'تونس', all_text)
        
        # Extract numbers using regex
        numbers = re.findall(r'\d+', all_text)
        
        if tunisia_pattern and numbers:
            # Format as "تونس number1 number2"
            formatted_plate = f"تونس {' '.join(numbers)}"
            return formatted_plate
        elif numbers:
            # If "تونس" not found but we have numbers
            return f"Plate: {' '.join(numbers)}"
        else:
            return "Unknown plate"
    
    def draw_results(self, img, results):
        """Draw detection results on the image"""
        result_img = img.copy()
        
        for plate_text, (x, y, w, h), ocr_result in results:
            # Draw rectangle around the license plate
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Put text above the rectangle
            cv2.putText(result_img, plate_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw OCR text locations if available
            for (bbox, text, prob) in ocr_result:
                # Convert relative coordinates to absolute
                pts = np.array([[int(p[0])+x, int(p[1])+y] for p in bbox])
                
                # Draw polygon around text
                cv2.polylines(result_img, [pts], True, (255, 0, 0), 2)
                
        return result_img
    
    def process_image(self, img_path):
        """Process a single image"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image at {img_path}")
            return None
            
        plate_regions = self.detect_plate(img)
        if not plate_regions:
            print("No license plates detected")
            return img
            
        results = self.recognize_text(img, plate_regions)
        result_img = self.draw_results(img, results)
        
        # Display summary at the top of the image
        summary = "Detected: " + " | ".join([r[0] for r in results])
        cv2.putText(result_img, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                   
        return result_img
    
    def process_video(self, video_source=0):
        """Process video from camera or file"""
        cap = cv2.VideoCapture(video_source)
        
        cap.set(3, 640)  # width
        cap.set(4, 480)  # height
        
        while True:
            success, img = cap.read()
            if not success:
                break
                
            plate_regions = self.detect_plate(img)
            
            if plate_regions:
                results = self.recognize_text(img, plate_regions)
                img = self.draw_results(img, results)
                
                # Show individual plate ROIs
                for _, (x, y, w, h), _ in results:
                    roi = img[y:y+h, x:x+w]
                    cv2.imshow("License Plate", roi)
            
            cv2.imshow("Tunisian Plate Detector", img)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
def main():
    parser = argparse.ArgumentParser(description='Tunisian License Plate Detection')
    parser.add_argument('-i', '--image', help='Path to image file')
    parser.add_argument('-v', '--video', help='Path to video file (omit for webcam)')
    
    args = parser.parse_args()
    
    detector = TunisianPlateDetector()
    
    if args.image:
        result_img = detector.process_image(args.image)
        if result_img is not None:
            cv2.imshow("Result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Use webcam or video file
        video_source = 0 if args.video is None else args.video
        detector.process_video(video_source)
        
if __name__ == "__main__":
    main()
