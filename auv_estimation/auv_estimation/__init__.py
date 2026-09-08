"""Layer 2: state estimation.

Fuses the Layer 1 gate outputs into one continuous estimate of where the
vehicle is and how fast it is going, published as nav_msgs/Odometry on
/odom/filtered, plus an honest report of how much that estimate is worth.

The filter itself is robot_localization, configured in config/ekf.yaml. The
only code in this package is nav_status, which answers the question
robot_localization does not: is this estimate currently aided or coasting.
"""
