#!/usr/bin/env python3
"""
Pose Localizer 3D ROS 2 Launch File
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    Generate launch description for 3D pose localization node.
    """
    
    # Launch arguments
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Input image topic'
    )
    
    model_complexity_arg = DeclareLaunchArgument(
        'model_complexity',
        default_value='1',
        description='MediaPipe model complexity: 0=lite, 1=full, 2=heavy'
    )
    
    # Pose Localizer Node
    pose_localizer_node = Node(
        package='f_formation_detection',
        executable='pose_localizer_3d_node',
        name='pose_localizer_3d',
        output='screen',
        parameters=[
            {
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5,
                'model_complexity': LaunchConfiguration('model_complexity'),
                'smooth_landmarks': True,
                'enable_segmentation': False,
                'publish_all_landmarks': False,
                'input_image_topic': LaunchConfiguration('image_topic'),
                'output_poses_topic': '/human_poses_3d',
            }
        ],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('image_topic')),
        ]
    )
    
    return LaunchDescription([
        image_topic_arg,
        model_complexity_arg,
        pose_localizer_node,
    ])
