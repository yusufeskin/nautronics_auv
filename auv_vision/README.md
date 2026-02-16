# AUV Vision (auv_vision)

The `auv_vision` package handles all computer vision tasks, including object detection, keypoint extraction, and image processing.

## Nodes

### `object_keypoint_detector.py`

Detects objects and their keypoints from the camera stream using deep learning models (e.g., YOLO).

- **Subscribes**: `/camera/image_raw` (sensor_msgs/Image)
- **Publishes**: Detected objects and keypoints (custom messages).

## Dependencies

- **OpenCV**: Computer vision library.
- **Ultralytics (YOLO)**: Real-time object detection.
- **`cv_bridge`**: Converts between ROS image messages and OpenCV images.

## Usage

To run the object detector:

```bash
ros2 run auv_vision object_keypoint_detector
```
