# Estimator (Layer 2) self-report, published alongside the filtered odometry.
#
# The mission layer uses `mode` and `dead_reckoning_time_s` to decide whether
# a manoeuvre is still safe to attempt; the controller uses `saturated`-style
# reasoning on the sigmas to decide whether to slow down.

std_msgs/Header header

uint8 MODE_INIT=0            # filter has not converged / not initialised
uint8 MODE_AIDED=1           # velocity aiding available (DVL bottom lock)
uint8 MODE_DEAD_RECKONING=2  # no velocity aiding, coasting on IMU + depth
uint8 MODE_FAULT=3           # estimate is not trustworthy, do not navigate
uint8 mode

bool dvl_fused
bool depth_fused
bool landmark_fused

# Seconds since the last velocity aiding update. Zero while aided. This is
# the number that decides how far a DVL dropout may be coasted through.
float32 dead_reckoning_time_s

float32 position_sigma_xy_m
float32 position_sigma_z_m
float32 yaw_sigma_rad

# Current gyro bias estimate [rad/s]. Exposed because a drifting bias is the
# dominant yaw error source once the magnetometer is out of the loop.
geometry_msgs/Vector3 gyro_bias
