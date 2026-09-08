"""Layer 4: turning a commanded wrench into vehicle motion.

ArduSub keeps the inner loop - attitude stabilisation and depth hold - and
this package owns the boundary to it. Everything above works in forces and
torques and knows nothing about MAVLink.

The package is named auv_actuation rather than auv_control only because the
latter is taken by the legacy stack. Rename it once that is retired.
"""
