ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30

rosdep install --from-paths src --ignore-src -r -y