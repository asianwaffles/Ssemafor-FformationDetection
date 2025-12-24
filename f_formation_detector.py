from enum import Enum
import numpy as np
import cv2
from pose_extractor import PoseExtractor


class FormationType(Enum):
    VIS_A_VIS = "vis-a-vis"    # Face-to-face
    L_SHAPED = "L-shaped"       # 90-degree arrangement
    SIDE_BY_SIDE = "side-by-side"
    CIRCULAR = "circular"
    NONE = "none"


class FFormationDetector:
    def __init__(self, use_3d=False):
        """
        Initialize F-formation detector.
        
        Args:
            use_3d: If True, use MediaPipe 3D poses. If False, use 2D bounding boxes.
        """
        self.use_3d = use_3d
        self.distance_threshold = 1.5 if use_3d else 300  # 1.5m for 3D, 300px for 2D
        self.angle_threshold = 60
        
        # Initialize 3D pose extractor if needed
        self.pose_extractor = None
        if use_3d:
            self.pose_extractor = PoseExtractor()

    def estimate_body_orientation_2d(self, bbox, image):
        """
        Estimate body orientation from 2D image (pixel variance).
        Returns 0=back, 1=front.
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        person_img = image[y1:y2, x1:x2]
        
        if person_img.size == 0:
            return 0.5
        
        if person_img.shape[0] < 10 or person_img.shape[1] < 10:
            return 0.5
        
        gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
        upper_third = gray[:max(1, gray.shape[0]//3), :]
        upper_var = np.var(upper_third)
        
        return min(upper_var / 1000.0, 1.0)

    def is_facing_direction(self, body_yaw, target_yaw, tolerance=45):
        """Check if body yaw is facing toward target direction."""
        angle_diff = abs(body_yaw - target_yaw)
        angle_diff = min(angle_diff, 360 - angle_diff)
        return angle_diff < tolerance

    def detect_formation_3d(self, positions_3d, landmarks):
        """
        Detect F-formation using 3D poses.
        
        Args:
            positions_3d: Array of 3D positions (N, 3) in meters
            landmarks: List of MediaPipe landmark arrays
        
        Returns:
            FormationType
        """
        if len(positions_3d) < 2:
            return FormationType.NONE
        
        # Get body orientations
        body_yaws = [self.pose_extractor.get_body_orientation_3d(lm) for lm in landmarks]
        body_yaws = np.array(body_yaws)
        
        distances = np.linalg.norm(positions_3d[:, None] - positions_3d, axis=2)
        
        # Two people
        if len(positions_3d) == 2:
            d = distances[0][1]
            print(f"  3D Distance: {d:.2f}m, Yaws: {body_yaws[0]:.1f}°, {body_yaws[1]:.1f}°")
            
            if d < self.distance_threshold:
                p1_to_p2 = positions_3d[1] - positions_3d[0]
                interaction_angle = np.degrees(np.arctan2(p1_to_p2[0], p1_to_p2[2]))
                
                p1_facing_p2 = self.is_facing_direction(body_yaws[0], interaction_angle, tolerance=60)
                p2_facing_p1 = self.is_facing_direction(body_yaws[1], interaction_angle + 180, tolerance=60)
                
                angle_between_yaws = abs(body_yaws[0] - body_yaws[1])
                angle_between_yaws = min(angle_between_yaws, 360 - angle_between_yaws)
                
                # Vis-a-vis: facing each other
                if p1_facing_p2 and p2_facing_p1:
                    print(f"  → Vis-a-vis formation (face-to-face)")
                    return FormationType.VIS_A_VIS
                
                # Side-by-side: similar yaws
                elif angle_between_yaws < 45:
                    print(f"  → Side-by-side formation")
                    return FormationType.SIDE_BY_SIDE
                
                # L-shaped: perpendicular
                elif 45 < angle_between_yaws < 135:
                    print(f"  → L-shaped formation")
                    return FormationType.L_SHAPED
                else:
                    return FormationType.SIDE_BY_SIDE
            
            return FormationType.NONE

        # Three or more people
        elif len(positions_3d) >= 3:
            max_distance = np.max(distances[distances > 0])
            print(f"  3D Max distance: {max_distance:.2f}m")
            
            if max_distance < self.distance_threshold:
                center = positions_3d.mean(axis=0)
                distances_to_center = np.linalg.norm(positions_3d - center, axis=1)
                std_dist = np.std(distances_to_center)
                mean_dist = np.mean(distances_to_center)
                
                # Circular: equidistant + facing inward
                if std_dist < mean_dist * 0.4:
                    facing_center_count = 0
                    for i, pos in enumerate(positions_3d):
                        to_center = center - pos
                        angle_to_center = np.degrees(np.arctan2(to_center[0], to_center[2]))
                        if self.is_facing_direction(body_yaws[i], angle_to_center, tolerance=60):
                            facing_center_count += 1
                    
                    if facing_center_count >= len(positions_3d) * 0.6:
                        print(f"  → Circular formation")
                        return FormationType.CIRCULAR
                
                print(f"  → Side-by-side formation (3+ people)")
                return FormationType.SIDE_BY_SIDE
            
            return FormationType.NONE

        return FormationType.NONE

    def detect_formation_2d(self, image, bboxes, positions):
        """Original 2D detection"""
        if len(bboxes) < 2:
            return FormationType.NONE
        
        orientations = [self.estimate_body_orientation_2d(bbox, image) for bbox in bboxes]
        distances = np.linalg.norm(positions[:, None] - positions, axis=2)
        
        if len(bboxes) == 2:
            d = distances[0][1]
            print(f"  2D Distance: {d:.1f}px")
            
            if d < self.distance_threshold:
                p1_to_p2 = positions[1] - positions[0]
                angle = np.degrees(np.arctan2(p1_to_p2[1], p1_to_p2[0]))
                
                if abs(angle) < 30 or abs(angle - 180) < 30:
                    print(f"  → Side-by-side formation")
                    return FormationType.SIDE_BY_SIDE
                elif abs(angle - 90) < 30 or abs(angle - 270) < 30:
                    print(f"  → Vis-a-vis formation")
                    return FormationType.VIS_A_VIS
                else:
                    print(f"  → L-shaped formation")
                    return FormationType.L_SHAPED
            
            return FormationType.NONE

        elif len(bboxes) >= 3:
            max_distance = np.max(distances[distances > 0])
            if max_distance < self.distance_threshold:
                center = positions.mean(axis=0)
                distances_to_center = np.linalg.norm(positions - center, axis=1)
                std_dist = np.std(distances_to_center)
                mean_dist = np.mean(distances_to_center)
                
                if std_dist < mean_dist * 0.5:
                    print(f"  → Circular formation")
                    return FormationType.CIRCULAR
                else:
                    print(f"  → Side-by-side formation (3+ people)")
                    return FormationType.SIDE_BY_SIDE
            
            return FormationType.NONE

        return FormationType.NONE

    def detect_formation(self, image, bboxes, positions, landmarks=None):
        """
        Main detection function - automatically selects 2D or 3D.
        
        Args:
            image: Input image
            bboxes: Bounding boxes (for 2D)
            positions: 2D positions OR 3D positions (Nx3)
            landmarks: MediaPipe landmarks (for 3D)
        
        Returns:
            FormationType
        """
        # Auto-detect if 3D data
        is_3d_data = len(positions.shape) > 1 and positions.shape[1] == 3 and positions[0][2] != 0
        
        if is_3d_data and landmarks is not None and self.pose_extractor:
            return self.detect_formation_3d(positions, landmarks)
        else:
            return self.detect_formation_2d(image, bboxes, positions)


def main():
    detector = FFormationDetector(use_3d=False)
    print("F-formation detector ready")


if __name__ == "__main__":
    main()
