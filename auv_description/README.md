# AUV Description (auv_description)

The `auv_description` package contains the physical description of the AUV, including 3D models and URDF/Xacro files.

## Overview

This package is used by the simulation environment (Gazebo) and for visualization in RViz. It defines the robot's links, joints, visual meshes, and collision geometries.

## Contents

- **`urdf/`**: Contains the `.xacro` files defining the robot model.
- **`models/`**: 3D mesh files (STL/DAE) for the visual representation.
- **`launch/`**: Launch files to spawn the robot in Gazebo or view it in RViz.

## Usage

To view the robot model in RViz:

```bash
ros2 launch auv_description view_robot.launch.py
```

To spawn the robot in a Gazebo simulation:

```bash
ros2 launch auv_description spawn_robot.launch.py
```
