#!/usr/bin/env python3
"""
F-Formation Detection Node for ROS 2 Humble.

This node processes camera images, detects people using YOLO, clusters them,
and determines their F-formation (social interaction formation).

Topics:
- Subscribes to: /camera/image_raw (sensor_msgs/Image)
- Publishes to: /formations (sensor_msgs/Image) - annotated image with detections
            and /group_formations (custom message) - formation data
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import warnings

from .f_formation_detector import FFormationDetector, FormationType

# Suppress warnings
warnings.filterwarnings('ignore')


class FormationDetectorNode(Node):
    def __init__(self):
        super().__init__('formation_detector')
        
        # Declare parameters
        self.declare_parameter('yolo_model', 'yolov8x.pt')
        self.declare_parameter('confidence_threshold', 0.20)
        self.declare_parameter('min_person_height_ratio', 0.03)
        self.declare_parameter('min_person_area_ratio', 0.0005)
        self.declare_parameter('aspect_ratio_min', 0.12)
        self.declare_parameter('aspect_ratio_max', 0.95)
        self.declare_parameter('dbscan_eps', 200)
        self.declare_parameter('dbscan_min_samples', 1)
        self.declare_parameter('input_image_topic', '/camera/image_raw')
        self.declare_parameter('output_image_topic', '/formation_detector/image_annotated')
        self.declare_parameter('use_3d_poses', False)
        
        # Get parameters
        yolo_model = self.get_parameter('yolo_model').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.min_person_height_ratio = self.get_parameter('min_person_height_ratio').value
        self.min_person_area_ratio = self.get_parameter('min_person_area_ratio').value
        self.aspect_ratio_min = self.get_parameter('aspect_ratio_min').value
        self.aspect_ratio_max = self.get_parameter('aspect_ratio_max').value
        self.dbscan_eps = self.get_parameter('dbscan_eps').value
        self.dbscan_min_samples = self.get_parameter('dbscan_min_samples').value
        input_topic = self.get_parameter('input_image_topic').value
        output_topic = self.get_parameter('output_image_topic').value
        self.use_3d = self.get_parameter('use_3d_poses').value
        
        # Initialize YOLO
        try:
            self.model = YOLO(yolo_model)
            self.model.conf = self.conf_threshold
            self.model.classes = [0]  # Only detect people
            self.get_logger().info(f'YOLO model loaded: {yolo_model}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            raise
        
        # Initialize F-formation detector
        try:
            self.f_formation_detector = FFormationDetector(use_3d=self.use_3d)
            self.get_logger().info('F-formation detector initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize F-formation detector: {e}')
            raise
        
        # Bridge for image conversion
        self.bridge = CvBridge()
        
        # Publishers
        self.image_pub = self.create_publisher(Image, output_topic, 10)
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )
        
        # Colors for groups
        self.colors = [
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue (BGR)
            (0, 0, 255),      # Red
            (255, 165, 0),    # Cyan
            (128, 0, 128),    # Purple
            (255, 255, 0),    # Yellow
            (0, 255, 255)     # Cyan
        ]
        
        self.get_logger().info(f'Subscribing to: {input_topic}')
        self.get_logger().info(f'Publishing to: {output_topic}')
        self.get_logger().info('Formation Detection Node ready')

    def filter_person_detections(self, box, img_shape):
        """Filter out false positives based on aspect ratio and size"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        width = x2 - x1
        height = y2 - y1
        
        aspect_ratio = width / height if height > 0 else 0
        min_height = img_shape[0] * self.min_person_height_ratio
        area = width * height
        min_area = img_shape[0] * img_shape[1] * self.min_person_area_ratio
        
        return (self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max and 
                height >= min_height and area >= min_area)

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info(f'📸 Received image: {img.shape}')
            
            # Run YOLO inference
            self.get_logger().info('🔍 Running YOLO detection...')
            results = self.model(img)
            self.get_logger().info('✅ YOLO detection complete')
            
            # Extract detections
            bboxes = []
            features = []
            
            for result in results:
                for box in result.boxes:
                    if (int(box.cls[0]) == 0 and 
                        box.conf[0].item() > self.conf_threshold and
                        self.filter_person_detections(box, img.shape)):
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0].item()
                        
                        x_center = (x1 + x2) // 2
                        y_center = (y1 + y2) // 2
                        features.append([x_center, y_center])
                        bboxes.append([x1, y1, x2, y2, confidence])
            
            if bboxes:
                features = np.array(features)
                bboxes = np.array(bboxes)
                
                self.get_logger().info(f'✨ Detected {len(bboxes)} people')
                
                # Perform DBSCAN clustering
                dbscan = DBSCAN(
                    eps=self.dbscan_eps,
                    min_samples=self.dbscan_min_samples,
                    metric='euclidean'
                ).fit(features)
                
                cluster_labels = dbscan.labels_
                num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                self.get_logger().info(f'👥 Found {num_clusters} groups')
                
                # Process each cluster
                for cluster_id in set(cluster_labels):
                    if cluster_id == -1:
                        continue
                    
                    cluster_mask = cluster_labels == cluster_id
                    cluster_positions = features[cluster_mask]
                    cluster_bboxes = bboxes[cluster_mask]
                    color = self.colors[cluster_id % len(self.colors)]
                    
                    try:
                        # Detect formation
                        formation = self.f_formation_detector.detect_formation(
                            img,
                            cluster_bboxes,
                            cluster_positions
                        )
                        
                        self.get_logger().debug(
                            f'Group {cluster_id+1}: {len(cluster_bboxes)} people, '
                            f'Formation: {formation.value}'
                        )
                        
                        # Draw annotations
                        for bbox in cluster_bboxes:
                            x1, y1, x2, y2, confidence = map(int, bbox)
                            
                            # Draw bounding box
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            
                            # Add labels
                            texts = [
                                f"G{cluster_id+1}",
                                f"{formation.value}",
                                f"{confidence:.2f}"
                            ]
                            
                            y_offset = y1 + 14
                            for text in texts:
                                text_size = cv2.getTextSize(
                                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
                                )[0]
                                cv2.rectangle(
                                    img, (x1, y_offset - 12),
                                    (x1 + text_size[0] + 4, y_offset + 3),
                                    (0, 0, 0), -1
                                )
                                cv2.putText(
                                    img, text, (x1 + 2, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
                                )
                                y_offset += 14
                    
                    except Exception as e:
                        self.get_logger().error(f'Error processing cluster {cluster_id}: {e}')
                        continue
            else:
                self.get_logger().info('⚠️ No people detected in this image')
            
            # Publish annotated image
            output_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            output_msg.header = msg.header
            self.image_pub.publish(output_msg)
            self.get_logger().info(f'📤 Published annotated image to /formation_detector/image_annotated')

        except Exception as e:
            self.get_logger().error(f'❌ Error in image callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = FormationDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
