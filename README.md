# F-Formation Detection - ROS 2 Humble
Note: These files were executed on a Virtual Machine, some stuff is hardcoded to a synced folder that I was using to transfer files into my VM.
Real-time social formation detection for robots using YOLO and DBSCAN clustering.

## Features

- 🎯 **YOLO Person Detection** - Fast and accurate person detection
- 👥 **DBSCAN Clustering** - Groups nearby people together
- �� **Formation Classification** - Detects formation types:
  - Vis-à-vis (face-to-face)
  - Side-by-side
  - L-shaped
  - Circular
  - None (no formation)
- 🤖 **ROS 2 Humble** - Full ROS 2 integration
- 📊 **Real-time Visualization** - Annotated output with boxes and labels

## Quick Start

### Installation

**Prerequisites:**
- Ubuntu 22.04 LTS
- ROS 2 Humble (installed and sourced)
- Python 3.10+

**Setup:**

```bash
# 1. Clone repository
cd /path/to/workspace
git clone https://github.com/YOUR_USERNAME/f-formation-detection.git
cd f-formation-detection

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Install NumPy 1.26.4 (requirements should install this but sometimes it bugs out idk why)
pip3 install 'numpy==1.26.4' --force-reinstall

# 4. Verify imports
python3 -c "from f_formation_detection.f_formation_detector import FFormationDetector; print('✅ Import successful')"
```

### Usage

**Terminal 1 - Detector Node:**

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=/path/to/f-formation-detection:$PYTHONPATH
python3 -m f_formation_detection.groupingimg
```

**Terminal 2 - Publish Images:**

Save this as `publish_images.py`:
```python
import cv2, rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pathlib import Path
import time

rclpy.init()
node = rclpy.create_node('image_publisher')
publisher = node.create_publisher(Image, '/camera/image_raw', 10)
bridge = CvBridge()

for image_path in sorted(Path('/media/sf_synced').glob('*.jpg')) + sorted(Path('/media/sf_synced').glob('*.png')):
    cv_image = cv2.imread(str(image_path))
    if cv_image is None:
        continue
    msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
    print(f"Publishing: {image_path.name}")
    publisher.publish(msg)
    time.sleep(2)
    rclpy.spin_once(node, timeout_sec=0.5)

print("Done!")
rclpy.shutdown()
```

Then run it:

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=/media/sf_synced:$PYTHONPATH
python3 publish_images.py
```

**Terminal 3 - Save Annotated Images:**

Save this as `save_images.py`:
```python
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pathlib import Path

rclpy.init()
node = rclpy.create_node('image_saver')
bridge = CvBridge()

output_dir = Path('/media/sf_synced/results')
output_dir.mkdir(exist_ok=True)
counter = [0]

def callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        output_path = output_dir / f"result_{counter[0]:03d}.jpg"
        cv2.imwrite(str(output_path), cv_image)
        print(f"💾 Saved: {output_path.name}")
        counter[0] += 1
    except Exception as e:
        print(f"Error: {e}")

subscription = node.create_subscription(
    Image,
    '/formation_detector/image_annotated',
    callback,
    10
)

print("Saving annotated images to /media/sf_synced/results/")
print("Press Ctrl+C to stop\n")

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    print("\n✅ Stopped saving images")
finally:
    rclpy.shutdown()
```

Then run it:

```bash
source /opt/ros/humble/setup.bash
python3 save_images.py
```

## ROS 2 Topics

**Subscribe:**
- `/camera/image_raw` (sensor_msgs/Image) - Input camera feed

**Publish:**
- `/formation_detector/image_annotated` (sensor_msgs/Image) - Annotated output with detections

## Package Structure

```
f_formation_detection/
├── __init__.py                    # Package initialization
├── f_formation_detector.py        # Core detection logic
└── groupingimg.py                 # ROS 2 node implementation

launch/
└── formation_detector_launch.py   # ROS 2 launch file

package.xml                        # ROS 2 manifest
setup.py                           # Python package setup
requirements.txt                   # Python dependencies
```

## Requirements

- rclpy>=0.13.0
- sensor-msgs>=0.0.0
- geometry-msgs>=0.0.0
- cv-bridge>=3.0.0
- opencv-python>=4.8.0
- ultralytics>=8.0.0
- numpy==1.26.4 (CRITICAL - not 2.x)
- scikit-learn>=1.0.0

## License

MIT License

## Support
For issues or questions, open an issue on GitHub.
