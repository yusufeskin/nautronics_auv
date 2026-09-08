# A pilot-input demand for ArduSub, in the units MANUAL_CONTROL expects.
#
# This is the boundary between the control stack and whoever owns the MAVLink
# connection. The control stack works in newtons and produces this; the
# autopilot bridge forwards it and does no thinking of its own.
#
# The split exists because only one process may hold the MAVLink link. Having
# the control stack open its own connection would mean two ports bound to the
# same endpoint and, worse, two independent sources of command authority over
# the same vehicle.

std_msgs/Header header

# Body axes as ArduSub's pilot input defines them, each in [-1000, 1000]:
#   x  forward positive
#   y  right positive
#   z  throttle stick; in ALT_HOLD the neutral value means "hold this depth"
#      rather than "cut thrust", and higher ascends
#   r  yaw, clockwise positive
#
# Note these are not one consistent frame: x and z follow the stick, while y
# and r invert against the FLU convention used everywhere else in this stack.
# The conversion happens once, in auv_actuation/demand_mapping.py.
int16 x
int16 y
int16 z
int16 r

# MANUAL_CONTROL button bitmask. Zero unless something needs a joystick
# button emulated.
uint16 buttons

# True when the requested wrench had to be clipped to fit these axes. The
# controller upstream holds its integrators while this is set.
bool saturated
