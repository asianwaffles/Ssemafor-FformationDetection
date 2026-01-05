"""F-Formation Detection ROS 2 Package"""

# Core detection logic (no ROS 2 dependencies)
from .f_formation_detector import FFormationDetector, FormationType

# ROS 2 nodes (optional - only import if rclpy is available)
try:
    from .pose_extractor import PoseLocalizer3DNode
    from .groupingimg import FormationDetectorNode
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False
    PoseLocalizer3DNode = None
    FormationDetectorNode = None

__all__ = [
    'FFormationDetector',
    'FormationType',
    'PoseLocalizer3DNode',
    'FormationDetectorNode',
]
