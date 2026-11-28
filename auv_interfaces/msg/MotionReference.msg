# Guidance -> position controller: where the vehicle should be now, plus the
# velocity and acceleration of that reference for feedforward.
# Everything is expressed in header.frame_id (odom); none of it is body frame.
std_msgs/Header header
geometry_msgs/Pose pose
geometry_msgs/Twist velocity       # linear [m/s], angular.z = yaw rate [rad/s]
geometry_msgs/Accel acceleration   # linear [m/s^2]
