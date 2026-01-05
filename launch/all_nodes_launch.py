#!/usr/bin/env python3
"""
Complete F-Formation Detection System Launch File
Runs both pose localization and formation detection
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    Generate launch description for complete system.
    """
    
    # Launch arguments
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Input image topic'
    )
    
    yolo_model_arg = DeclareLaunchArgument(
        'yolo_model',
        default_value='yolov8x.pt',
        description='YOLO model to use'
    )
    
    use_3d_arg = DeclareLaunchArgument(
        'use_3d_poses',
        default_value='false',
        description='Use 3D pose estimation'
    )
    
    # Pose Localizer Node (optional)
    pose_localizer_node = Node(
        package='f_formation_detection',
        executable='pose_localizer_3d_node',
        name='pose_localizer_3d',
        output='screen',
        parameters=[
            {
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5,
                'model_complexity': 1,
                'smooth_landmarks': True,
                'enable_segmentation': False,
                'publish_all_landmarks': False,
                'input_image_topic': LaunchConfiguration('image_topic'),
                'output_poses_topic': '/human_poses_3d',
            }
        ],
    )
    
    # Formation Detector Node
    formation_detector_node = Node(
        package='f_formation_detection',
        executable='formation_detector_node',
        name='formation_detector',
        output='screen',
        parameters=[
            {
                'yolo_model': LaunchConfiguration('yolo_model'),
                'confidence_threshold': 0.20,
                'min_person_height_ratio': 0.03,
                'min_person_area_ratio': 0.0005,
                'aspect_ratio_min': 0.12,
                'aspect_ratio_max': 0.95,
                'dbscan_eps': 200,
                'dbscan_min_samples': 1,
                'input_image_topic': LaunchConfiguration('image_topic'),
                'output_image_topic': '/formation_detector/image_annotated',
                'use_3d_poses': LaunchConfiguration('use_3d_poses'),
            }
        ],
    )
    
    return LaunchDescription([
        image_topic_arg,
        yolo_model_arg,
        use_3d_arg,
        # Only run formation detector by default
        # Uncomment pose_localizer_node to enable 3D pose estimation
        formation_detector_node,
    ])
