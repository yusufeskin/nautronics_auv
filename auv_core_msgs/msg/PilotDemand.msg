# Pilot input for ArduSub, as six RC channels in PWM microseconds.
#
# This is the boundary between the control stack and whoever owns the MAVLink
# connection. The output stage in auv_actuation produces it; the autopilot
# bridge forwards it as RC_CHANNELS_OVERRIDE and does no thinking of its own.
#
# The split exists because only one process may hold the MAVLink link. Having
# the control stack open its own connection would mean two ports bound to the
# same endpoint and, worse, two independent sources of command authority over
# the same vehicle.
#
# Channel numbering and directions are ArduSub's own, not this stack's FLU.
# 1500 is neutral on every channel; the range is 1100..1900. The conversion
# from FLU happens once, in auv_actuation/demand_mapping.py.

std_msgs/Header header

uint16 pitch     # ch1, lean angle; neutral holds level
uint16 roll      # ch2, lean angle; neutral holds level
uint16 throttle  # ch3, climb rate, higher ascends; neutral holds depth
uint16 yaw       # ch4, turn rate, higher turns clockwise; neutral holds heading
uint16 forward   # ch5, thrust, higher is forward
uint16 lateral   # ch6, thrust, higher is right
