# One observation of a known competition object, expressed in the frame of
# the sensor that saw it (header.frame_id, e.g. "front_camera_link").
#
# This is the Layer 3 -> Layer 2 contract: perception publishes these, the
# landmark tracker turns them into map entries, and the map->odom correction
# derived from them is what bounds yaw and XY drift over a run. Nothing here
# is expressed in a world frame on purpose - perception must never need to
# know where the vehicle thinks it is.

std_msgs/Header header

# Object class from the competition task list, e.g. "gate", "path_marker",
# "slalom_pipe", "bin", "torpedo_board", "octagon".
string class_id

# Stable identifier when the same physical object is re-observed across
# frames; empty when the detector cannot tell instances apart.
string instance_id

# Landmark pose in header.frame_id, with the detector's uncertainty.
geometry_msgs/PoseWithCovariance pose

# Detector confidence in [0, 1].
float32 confidence

# False when only position (or bearing) is observable and the orientation
# component of `pose` is meaningless - a single round buoy, for example.
# Consumers must check this before using the orientation to correct yaw.
bool has_orientation
