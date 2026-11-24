# Nautronics AUV

**Nautronics AUV** is a ROS 2 based Autonomous Underwater Vehicle project. This repository contains the source code for simulation, control, vision, and hardware integration.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Packages](#packages)
- [Credits](#credits)

## Requirements

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 (Jammy) |
| ROS 2 | Humble (`ros-humble-desktop`) |
| Gazebo | Harmonic (gz-sim 8) |
| ArduPilot | `master`, ArduSub SITL |
| [ArduPilot Gazebo plugin](https://github.com/ArduPilot/ardupilot_gazebo) | `main`, from source |
| [ros_gz](https://github.com/gazebosim/ros_gz) | `humble` branch, **built from source against Harmonic** |

**Set these up first by following [docs/sim_setup.md](docs/sim_setup.md).** Each step there ends with a check.

> ⚠️ Do not install the `ros-humble-ros-gz*` binary packages or `ros-humble-desktop-full`. They are built for Gazebo Fortress and break the Harmonic setup. The guide explains why.

## Installation

With the [simulation environment](docs/sim_setup.md) ready, build our workspace:

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

   Add it to the end of your `~/.bashrc`, **after** the `ros_gz_ws` line from the setup guide:
   ```bash
   echo 'source ~/nautronics_ws/install/setup.bash' >> ~/.bashrc
   source ~/.bashrc
   ```

The `auv_description` models are added to `GZ_SIM_RESOURCE_PATH` automatically by its environment hook when you source the workspace. The ArduPilot Gazebo paths are set in [step 4 of the setup guide](docs/sim_setup.md#4-ardupilot-gazebo-plugin).

## Usage

### Simulation

Start Gazebo with the vehicle:

```bash
ros2 launch auv_description gazebo.launch.py                          # default world: robosub_2026.world
ros2 launch auv_description gazebo.launch.py world:=teknofest_pool.world
```

This starts Gazebo, the ROS <-> Gazebo bridge, the vehicle (plus the two torpedo props) and the simulated DVL. Add `dvl:=false` to skip the DVL.

See [auv_description](auv_description/README.md) for the other worlds and launch arguments.

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
