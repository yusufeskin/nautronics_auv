# Nautronics AUV

**Nautronics AUV** is a ROS 2 based Autonomous Underwater Vehicle project. This repository contains the source code for simulation, control, vision, and hardware integration.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Packages](#packages)
- [Credits](#credits)

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

Add the necessary model paths to your `~/.bashrc` to ensure Gazebo can find the AUV models:

```bash
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$HOME/nautronics_auv/src/auv_description/models:$HOME/nautronics_auv/src/auv_description/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
source ~/.bashrc
```

## Packages

This repository consists of several ROS 2 packages:

- **[auv_bt](auv_bt/README.md)**: Behavior Tree implementations for mission planning.
- **[auv_cam](auv_cam/README.md)**: Camera handling and image collection nodes.
- **[auv_control](auv_control/README.md)**: Control algorithms, including thruster mixing and visual servoing.
- **[auv_description](auv_description/README.md)**: URDF/Xacro descriptions and simulation models.
- **[auv_hardware](auv_hardware/README.md)**: Hardware interfaces for sensors and actuators.
- **[auv_interfaces](auv_interfaces/README.md)**: Custom ROS 2 messages and service definitions.
- **[auv_navigation](auv_navigation/README.md)**: Navigation stack and path planning.
- **[auv_vision](auv_vision/README.md)**: Computer vision algorithms (Object detection, Keypoint detection).
- **[additional_drivers](additional_drivers/bno055/README.md)**: External drivers (e.g., BNO055 IMU).

## Credits

The RoboSub task models and the course layout in the simulation are **not
ours** - they are the [ITU AUV Team](https://github.com/itu-auv/auv-software)'s
work, redistributed here under their BSD 3-Clause licence with our thanks.
That covers the gate, slalom, path markers, bins, torpedo board, octagon and
the Task 5 pickup objects in `auv_description/models/`, and the layout in
`auv_description/worlds/robosub_2026.world`. We adapted them to run under
Gazebo Sim; those changes are listed in
[`auv_description/models/ITU_AUV_NOTICE`](auv_description/models/ITU_AUV_NOTICE).

The pool textures come from
[HonuRobotics/bluerobotics_models](https://github.com/HonuRobotics/bluerobotics_models)
(Apache-2.0).

See [NOTICE](NOTICE) for the full list of third-party assets and their
licences.
