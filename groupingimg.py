import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import os
import warnings
from f_formation_detector import FFormationDetector, FormationType

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize detectors
try:
    model = YOLO("yolov8x.pt")  # Use larger model for better detection
    model.conf = 0.20  # Lower threshold for better detection
    model.classes = [0]  # Only detect people
    f_formation_detector = FFormationDetector()
    print("Models initialized successfully")
except Exception as e:
    print(f"Error initializing models: {e}")
    exit(1)

# Configure paths
DATASET_PATH = "/Users/alanlin/Downloads/MPII Dataset Batch"
OUTPUT_PATH = os.path.join(DATASET_PATH, "output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

def filter_person_detections(box, img_shape):
    """Filter out false positives like balls based on aspect ratio and size"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # More tolerant aspect ratio for moving/blurry people and edge cases
    aspect_ratio = width / height if height > 0 else 0
    
    # Lower minimum height threshold for distant/blurry/edge people
    min_height = img_shape[0] * 0.03  # Reduced from 0.05 to catch edge people
    
    # Area check to distinguish from small objects
    area = width * height
    min_area = img_shape[0] * img_shape[1] * 0.0005  # Reduced from 0.001
    
    # Widened aspect ratio tolerance (0.12 to 0.95)
    return (0.12 <= aspect_ratio <= 0.95) and (height >= min_height) and (area >= min_area)

def process_image(image_path):
    """Process a single image for person detection and F-formation analysis"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Run YOLO inference
    results = model(img)

    # Initialize lists
    bboxes = []
    features = []
    
    for result in results:
        for box in result.boxes:
            # Only process if it's a person (class 0) and passes size/ratio filter
            if (int(box.cls[0]) == 0 and 
                box.conf[0].item() > 0.20 and  # Lowered from 0.25 to catch edge people
                filter_person_detections(box, img.shape)):
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                
                # Compute center point
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                features.append([x_center, y_center])
                bboxes.append([x1, y1, x2, y2, confidence])
                print(f"Detected person at ({x_center}, {y_center}) with confidence {confidence:.2f}")

    if not bboxes:
        print("No people detected in image")
        return

    # Convert to numpy arrays
    features = np.array(features)
    bboxes = np.array(bboxes)

    print(f"Total people detected: {len(bboxes)}")

    # Perform DBSCAN clustering
    dbscan = DBSCAN(
        eps=200,
        min_samples=1,
        metric='euclidean'
    ).fit(features)
    
    cluster_labels = dbscan.labels_
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of groups: {num_clusters}")

    # Define colors for groups
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
             (255, 165, 0), (128, 0, 128), (255, 255, 0), (0, 255, 255)]

    # Process each cluster
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue
            
        # Get cluster members
        cluster_mask = cluster_labels == cluster_id
        cluster_positions = features[cluster_mask]
        cluster_bboxes = bboxes[cluster_mask]
        
        # Get color for this cluster
        color = colors[cluster_id % len(colors)]
        
        try:
            # Detect formation
            formation = f_formation_detector.detect_formation(
                img,
                cluster_bboxes,
                cluster_positions
            )
            
            print(f"Group {cluster_id+1}: {len(cluster_bboxes)} people, Formation: {formation.value}")
            
            # Add formation debug
            if formation == FormationType.NONE:
                print(f"  → No formation detected (likely insufficient head pose data)")
            
                        # Draw detections and labels
            for bbox in cluster_bboxes:
                x1, y1, x2, y2, confidence = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add SMALLER labels INSIDE the box at the top
                texts = [
                    f"G{cluster_id+1}",
                    f"{formation.value}",
                    f"{confidence:.2f}"
                ]
                
                # Draw text inside box with smaller font
                y_offset = y1 + 14
                for text in texts:
                    # Add black background rectangle for text
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                    cv2.rectangle(img, (x1, y_offset - 12), 
                                 (x1 + text_size[0] + 4, y_offset + 3), 
                                 (0, 0, 0), -1)
                    # Draw white text (smaller font size: 0.35 instead of 0.5)
                    cv2.putText(img, text, (x1 + 2, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    y_offset += 14
                    
        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save result
    output_path = os.path.join(OUTPUT_PATH, f"processed_{os.path.basename(image_path)}")
    if cv2.imwrite(output_path, img):
        print(f"Saved result to: {output_path}")
    else:
        print(f"Failed to save image: {output_path}")

def main():
    """Process all images in the dataset"""
    print(f"Looking for images in: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path not found: {DATASET_PATH}")
        return

    # Process all images
    image_count = 0
    for filename in os.listdir(DATASET_PATH):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_count += 1
            print(f"\n{'='*60}")
            print(f"Processing {image_count}: {filename}")
            print(f"{'='*60}")
            try:
                image_path = os.path.join(DATASET_PATH, filename)
                process_image(image_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue

if __name__ == "__main__":
    main()
