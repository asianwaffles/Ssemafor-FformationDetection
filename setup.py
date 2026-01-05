from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'f_formation_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'opencv-python',
        'mediapipe',
        'numpy',
        'scikit-learn',
        'ultralytics',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='F-formation detection for social robots using YOLO and MediaPipe',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_localizer_3d_node = f_formation_detection.pose_extractor:main',
            'formation_detector_node = f_formation_detection.groupingimg:main',
        ],
    },
)
