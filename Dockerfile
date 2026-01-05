FROM osrf/ros:humble-desktop

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p /root/ros2_ws/src
WORKDIR /root/ros2_ws/src

# Clone the package (modify URL as needed)
# RUN git clone https://github.com/yourusername/f_formation_detection.git

# For local development, copy the package
COPY . f_formation_detection/

WORKDIR /root/ros2_ws

# Install Python dependencies
RUN pip install -U \
    opencv-python \
    mediapipe \
    numpy \
    scikit-learn \
    ultralytics \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ROS dependencies
RUN apt-get update && rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# Build the package
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

# Setup bashrc
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'source /root/ros2_ws/install/setup.bash' >> /root/.bashrc && \
    echo 'export ROS_DOMAIN_ID=0' >> /root/.bashrc

# Default command
CMD ["bash"]
