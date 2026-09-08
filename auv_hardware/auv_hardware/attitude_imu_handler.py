"""Publishes the autopilot's own attitude as a sensor_msgs/Imu.

This is the vehicle's absolute heading source. ArduSub fuses its compass into
EKF3 and reports the result in ATTITUDE; this handler converts that into the
message heading_gate expects, and heading_gate then cross-checks it against
the external IMU's gyro before the estimator is allowed to use it.

Two conversions happen here and both matter.

**Frame.** ATTITUDE is Euler angles of the body FRD frame relative to a NED
world - the autopilot's convention. Every ROS topic in this stack is ENU/FLU.
The conversion is a composition of the two convention flips in
auv_core.frames, and doing it anywhere by hand is how a heading ends up
mirrored.

**Nothing else.** The orientation is passed on as a measurement, not as a
command. The control loop never reads it to decide where to point; it reads
the estimator, which has fused this with everything else. What the autopilot
believes about north and what this stack believes about north are allowed to
differ, and the estimator is where that gets reconciled.
"""

import math

import numpy as np
from sensor_msgs.msg import Imu

from auv_core import frames


class AttitudeImuHandler:
    def __init__(self, node, publisher, frame_id="heading_link",
                 orientation_stddev_deg=5.0, rate_stddev_deg_s=1.0):
        self.node = node
        self.publisher = publisher
        self.frame_id = frame_id

        orientation_var = math.radians(orientation_stddev_deg) ** 2
        rate_var = math.radians(rate_stddev_deg_s) ** 2
        self._orientation_cov = [orientation_var, 0.0, 0.0,
                                 0.0, orientation_var, 0.0,
                                 0.0, 0.0, orientation_var]
        self._rate_cov = [rate_var, 0.0, 0.0,
                          0.0, rate_var, 0.0,
                          0.0, 0.0, rate_var]

    def handle_message(self, msg):
        # ATTITUDE gives the body(FRD) -> world(NED) rotation as Euler angles.
        q_ned_frd = frames.quat_from_euler_rpy(msg.roll, msg.pitch, msg.yaw)
        # Flip both conventions to get body(FLU) -> world(ENU).
        q_enu_flu = frames.flip_world_quat(frames.flip_body_quat(q_ned_frd))

        # Body rates are FRD as well, so only the body flip applies.
        rates = frames.flip_body_convention(
            np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
        )

        out = Imu()
        out.header.stamp = self.node.get_clock().now().to_msg()
        out.header.frame_id = self.frame_id

        out.orientation.x = float(q_enu_flu[0])
        out.orientation.y = float(q_enu_flu[1])
        out.orientation.z = float(q_enu_flu[2])
        out.orientation.w = float(q_enu_flu[3])
        out.orientation_covariance = self._orientation_cov

        out.angular_velocity.x = float(rates[0])
        out.angular_velocity.y = float(rates[1])
        out.angular_velocity.z = float(rates[2])
        out.angular_velocity_covariance = self._rate_cov

        # ATTITUDE carries no specific force. A leading -1 is the
        # sensor_msgs/Imu convention for "this field is not provided", which
        # is more honest than publishing zeros that look like free-fall.
        out.linear_acceleration_covariance = [-1.0] + [0.0] * 8

        self.publisher.publish(out)
