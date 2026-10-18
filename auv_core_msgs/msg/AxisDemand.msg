# What the controller asks of the vehicle, per axis, for ArduSub in ALT_HOLD.
#
# This is the boundary between the control law and the output stage. The axes
# are deliberately not all forces: in ALT_HOLD the throttle and yaw sticks are
# rate demands that ArduSub closes itself, so that is what is carried here.
# The output stage turns this into RC channel PWM; nothing upstream of it
# knows about sticks.
#
# All in this stack's body frame, FLU.

std_msgs/Header header

# Fraction of full horizontal thrust, each in [-1, 1].
float32 surge       # forward positive
float32 sway        # left positive

# Rate demands, closed by ArduSub.
float32 climb_rate  # m/s, up positive. Zero means hold the current depth.
float32 yaw_rate    # rad/s, counter-clockwise positive. Zero means hold heading.
