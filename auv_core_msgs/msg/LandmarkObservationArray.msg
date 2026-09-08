# All landmark observations extracted from a single sensor frame.
#
# Published even when empty, so a consumer can tell "the detector ran and
# saw nothing" from "the detector is not running".

std_msgs/Header header
LandmarkObservation[] observations
