# Output of thrust allocation (Layer 5): the per-thruster demand that
# realises a commanded body wrench.
#
# Ordering matches the `thrusters` list in auv_core/config/vehicle.yaml.

std_msgs/Header header

# Commanded force per thruster in newtons, positive along the thruster's
# configured axis.
float32[] thrust_n

# Same command after the thruster force->PWM curve is applied. Empty when
# the curve has not been calibrated yet.
uint16[] pwm_us

# True when allocation had to clip at least one thruster to stay inside its
# force limits, meaning the realised wrench differs from the commanded one.
# Controllers should stop integrating while this is set.
bool saturated
