# Nautronics AUV

**Nautronics AUV** is a ROS 2 based Autonomous Underwater Vehicle project. This repository contains the source code for simulation, control, vision, and hardware integration.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Packages](#packages)

## Requirements

- **ROS 2** (Humble Hawksbill recommended)
- **ArduPilot**
  - [https://github.com/ArduPilot/ardupilot](https://github.com/ArduPilot/ardupilot)
- **ArduPilot Gazebo**  
  - [https://github.com/ArduPilot/ardupilot_gazebo](https://github.com/ArduPilot/ardupilot_gazebo)
- **Gazebo Sim (Harmonic for ROS 2)**  
  - [https://gazebosim.org/docs/latest/getstarted/](https://gazebosim.org/docs/latest/getstarted/)

## Installation

Follow these steps to set up the development environment:

1. **Create a Workspace**
   ```bash
   mkdir -p ~/nautronics_ws/src
   cd ~/nautronics_ws/src
   ```

2. **Clone the Repository**
   ```bash
   git clone https://github.com/yusufeskin/nautronics_auv
   cd ~/nautronics_ws
   ```

3. **Import Dependencies**
   ```bash
   vcs import src < src/nautronics_auv/nautronics.repos
   ```

4. **Install Python Requirements**
   ```bash
   pip3 install -r src/nautronics_auv/requirements.txt
   ```

5. **Install ROS Dependencies**
   ```bash
   sudo apt update
   rosdep update
   rosdep install --from-paths src --ignore-src -r -y
   ```

6. **Build the Workspace**
   ```bash
   colcon build --symlink-install
   ```

7. **Source the Workspace**
   ```bash
   source install/setup.bash
   ```

### GZ Sim & ArduPilot Configuration

Add the ArduPilot Gazebo model paths to your `~/.bashrc`:

```bash
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
source ~/.bashrc
```

The `auv_description` models are added to `GZ_SIM_RESOURCE_PATH` automatically by its environment hook when you run `source install/setup.bash`.

## Packages

This repository consists of several ROS 2 packages:

- **[auv_bt](auv_bt/README.md)**: Behavior Tree implementations for mission planning.
- **[auv_cam](auv_cam/README.md)**: Camera handling and image collection nodes.
- **[auv_control](auv_control/README.md)**: Control algorithms, including thruster mixing and visual servoing.
- **[auv_description](auv_description/README.md)**: URDF/Xacro descriptions and simulation models.
- **[auv_hardware](auv_hardware/README.md)**: Hardware interfaces for sensors and actuators.
- **[auv_interfaces](auv_interfaces/README.md)**: Custom ROS 2 messages and service definitions.
- **[auv_vision](auv_vision/README.md)**: Computer vision algorithms (Object detection, Keypoint detection).
- **additional_drivers**: External drivers imported via `nautronics.repos` ([gscam2](https://github.com/clydemcqueen/gscam2) camera driver, [waterlinked_dvl](https://github.com/yusufeskin/waterlinked_dvl) DVL driver).