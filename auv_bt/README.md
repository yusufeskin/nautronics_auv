# AUV Behavior Tree (auv_bt)

The `auv_bt` package implements the high-level mission control using Behavior Trees (BTs). It leverages the `py_trees` and `py_trees_ros` libraries to create modular and reactive behaviors for the AUV.

## Overview

Behavior Trees allow for complex mission planning by combining simple behaviors into a tree structure. This package organizes these behaviors and defines specific missions.

## Dependencies

- **py_trees**: Core behavior tree library.
- **py_trees_ros**: ROS 2 extensions for `py_trees`.

## Structure

- **`common_behaviors`**: Contains reusable behavior definitions that can be shared across different missions.
- **`gate`**: Specific mission logic for the gate task.

## Usage

To run a mission node (example):

```bash
ros2 run auv_bt mission_node
```

## Adding New Behaviors

1. Create a new behavior class inheriting from `py_trees.behaviour.Behaviour`.
2. Implement the `initialise()`, `update()`, and `terminate()` methods.
3. Register the behavior in the tree.
