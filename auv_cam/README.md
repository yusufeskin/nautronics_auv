# AUV Camera (auv_cam)

The `auv_cam` package handles camera operations and image data collection for the AUV.

## Overview

This package provides nodes to interface with the camera, capture images, and publish them to ROS topics for processing by other packages like `auv_vision`.

## Nodes

### `image_collector.py`

This node is responsible for collecting images from the camera stream, typically for creating datasets or logging.

#### Topics

- **Subscribes**:
  - `/camera/image_raw` (sensor_msgs/Image): The raw image stream from the camera.

- **Publishes**:
  - *(Topic details would go here if applicable)*

## Usage

To run the image collector node:

```bash
ros2 run auv_cam image_collector
```
