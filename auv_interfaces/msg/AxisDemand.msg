# Normalised motion demand: Jetson position controller -> RC output stage.
#
# Body frame, FLU (REP-103). Every field is in [-1, 1]; 0 means "stick centred".
# The four axes are NOT the same kind of quantity, because of how ArduSub's
# ALT_HOLD interprets each RC channel:
#
#   surge, sway : thrust fraction   (ch5, ch6)   -> 0 = no horizontal thrust
#   heave       : climb-rate demand (ch3)        -> 0 = ArduSub holds this depth
#   yaw         : turn-rate demand  (ch4)        -> 0 = ArduSub holds this heading
#
# That is why this is not a geometry_msgs/Twist: a Twist claims velocities on
# every axis, which is not what reaches the vehicle.
std_msgs/Header header

float64 surge   # + forward
float64 sway    # + left
float64 heave   # + up
float64 yaw     # + counter-clockwise seen from above
