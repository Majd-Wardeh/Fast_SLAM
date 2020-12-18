# FAST SLAM

This repo contains the implementation of Fast SLAM algorithm in Probabilistic Robotics book. Implemented and modified by Majd Wardeh.

## Installation

### Requirements

The code was tested with Ubuntu 18.04 and ROS Melodic.

### Step-by-Step Procedure
```bash
mkdir -p catkin_ws/src
cd catkin_ws
catkin init
catkin config --extend /opt/ros/$ROS_VERSION
catkin config --merge-devel
cd src
git clone 

# create virtual envirnment called fast_slam_ven
cd ~
virtualenv -p python2.7 ./fast_slam_ven
source ./fast_slam_ven/bin/activate

pip2 install -r python_dependencies.txt
