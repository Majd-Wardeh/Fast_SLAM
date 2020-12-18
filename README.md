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
cd src
git clone https://github.com/Majd-Wardeh/Fast_SLAM.git
cd ..
catkin_make

# create virtual envirnment called fast_slam_ven
cd ~
virtualenv -p python2.7 ./fast_slam_ven
source ./fast_slam_ven/bin/activate
cd catkin_ws/src/Fast_SLAM/
pip2 install -r python_dependencies.txt
```

## Testing
Open a terminal and type:
```bash
source ~/catkin_ws/devel/setup.bash
roslaunch husky_gazebo husky_playpen.launch
```
Open another terminal and type:
```bash
source ~/catkin_ws/devel/setup.bash
source ~/fast_slam_ven/bin/activate
rosrun fast_slam fast_slam.py 
```
