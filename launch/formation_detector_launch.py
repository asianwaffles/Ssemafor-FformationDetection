#!/usr/bin/env python3
"""
Formation Detector ROS 2 Launch File
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    Generate launch description for formation detection nodes.
    """
    
    # Launch arguments
    yolo_model_arg = DeclareLaunchArgument(
        'yolo_model',
        default_value='yolov8x.pt',
        description='YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)'
    )
    
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Input image topic'
    )
    
    use_3d_arg = DeclareLaunchArgument(
        'use_3d_poses',
        default_value='false',
        description='Use 3D pose estimation (requires valid camera calibration)'
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
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('image_topic')),
        ]
    )
    
    return LaunchDescription([
        yolo_model_arg,
        image_topic_arg,
        use_3d_arg,
        formation_detector_node,
    ])
