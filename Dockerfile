FROM ros:noetic-ros-base-focal

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3-catkin-tools \
        python3-numpy \
        python3-pip \
        python3-scipy \
        python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

# Dependencies required by the POLARIS GEM e2 simulator.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-noetic-ackermann-msgs \
        ros-noetic-effort-controllers \
        ros-noetic-geometry2 \
        ros-noetic-hector-gazebo \
        ros-noetic-hector-models \
        ros-noetic-jsk-rviz-plugins \
        ros-noetic-ros-control \
        ros-noetic-ros-controllers \
        ros-noetic-velodyne-simulator \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fluxbox \
        mesa-utils \
        novnc \
        websockify \
        x11vnc \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.4 supports Ubuntu 20.04's Python 3.8. Pin its pure-Python
# dependencies because newer releases have dropped Python 3.8 support.
RUN python3 -m pip install --no-cache-dir \
    filelock==3.16.1 \
    fsspec==2024.10.0 \
    jinja2==3.1.4 \
    sympy==1.13.3 \
    typing-extensions==4.12.2 \
    && python3 -m pip install --no-cache-dir --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1

# CasADi has no Ubuntu Focal rosdep key. Install the explicitly pinned Python
# dependency without upgrading Ubuntu's ROS-compatible NumPy.
COPY requirements-control.txt /tmp/requirements-control.txt
RUN python3 -m pip install --no-cache-dir --no-deps \
    -r /tmp/requirements-control.txt

COPY --chmod=755 scripts/start-gui.sh /usr/local/bin/start-gui

WORKDIR /workspace

CMD ["bash"]
