#!/usr/bin/env python3
"""Gazebo -> ROS2 bridge for the simulated DVL sensor (gz-sim 'dvl' plugin).

ros_gz_bridge's stock parameter_bridge has no built-in factory for
gz.msgs.DVLVelocityTracking <-> marine_acoustic_msgs/msg/Dvl, so this node
subscribes to the raw Gazebo Transport topic directly (via the gz-transport
Python bindings) and republishes it as a marine_acoustic_msgs/Dvl message
shaped the same way the real WaterLinked A50 driver's output is shaped, so
downstream code (dvl_odom_handler.py, sim_dvl_adapter.py) doesn't need to
know whether it's talking to sim or real hardware.

Beam geometry (id -> rotation/tilt, degrees) mirrors
auv_description/models/prototype_vehicle/macros/dvl_macro.xacro - keep the
two in sync if the sensor mount ever changes.
"""
import math
import os

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from marine_acoustic_msgs.msg import Dvl

import gz.transport13 as gz_transport
from gz.msgs10.dvl_velocity_tracking_pb2 import DVLVelocityTracking

UNKNOWN_COVAR = -1.0

# rotation = azimuth around the DVL's +z (down) axis, tilt = angle from +z.
# Matches dvl_macro.xacro's <arrangement> table.
BEAM_GEOMETRY_DEG = {
    1: {"rotation": 45.0, "tilt": 22.5},
    2: {"rotation": 135.0, "tilt": 22.5},
    3: {"rotation": -135.0, "tilt": 22.5},
    4: {"rotation": -45.0, "tilt": 22.5},
}

# gz DVLVelocityTracking.DVLType -> marine_acoustic_msgs/Dvl dvl_type
GZ_TO_ROS_DVL_TYPE = {
    1: Dvl.DVL_TYPE_PISTON,          # DVL_TYPE_PISTON
    2: Dvl.DVL_TYPE_PHASED_ARRAY,    # DVL_TYPE_PHASED_ARRAY
}

# gz DVLTrackingTarget.TargetType -> marine_acoustic_msgs/Dvl velocity_mode
# (numeric values happen to line up: BOTTOM=1, WATER_MASS=2)
GZ_TO_ROS_VELOCITY_MODE = {
    1: Dvl.DVL_MODE_BOTTOM,
    2: Dvl.DVL_MODE_WATER,
}


def beam_unit_vector(beam_id):
    # Computed in gz's FLU sensor frame, then flipped to FRD below to match
    # out.velocity's convention (see _on_gz_dvl).
    geom = BEAM_GEOMETRY_DEG.get(beam_id)
    if geom is None:
        return Vector3(x=0.0, y=0.0, z=-1.0)
    rotation = math.radians(geom["rotation"])
    tilt = math.radians(geom["tilt"])
    return Vector3(
        x=math.sin(tilt) * math.cos(rotation),
        y=-math.sin(tilt) * math.sin(rotation),
        z=-math.cos(tilt),
    )


class SimDvlBridge(Node):

    def __init__(self):
        super().__init__('sim_dvl_bridge')

        self.declare_parameter('gz_topic', '/model/prototype_vehicle/dvl')
        self.declare_parameter('ros_topic', '/sim/dvl/raw')
        self.declare_parameter('frame_id', 'dvl_link')
        self.declare_parameter('min_good_beams', 3)

        self.frame_id = self.get_parameter('frame_id').value
        self.min_good_beams = self.get_parameter('min_good_beams').value
        gz_topic = self.get_parameter('gz_topic').value
        ros_topic = self.get_parameter('ros_topic').value

        self.pub = self.create_publisher(Dvl, ros_topic, 10)

        self._gz_node = gz_transport.Node()
        ok = self._gz_node.subscribe(DVLVelocityTracking, gz_topic, self._on_gz_dvl)
        if not ok:
            self.get_logger().error(f"gz-transport subscribe failed on '{gz_topic}'")
        else:
            self.get_logger().info(
                f"sim_dvl_bridge: '{gz_topic}' (gz) -> '{ros_topic}' (ROS) basladi"
            )

    def _on_gz_dvl(self, msg: DVLVelocityTracking):
        out = Dvl()
        out.header.frame_id = self.frame_id
        if msg.header.stamp.sec or msg.header.stamp.nsec:
            out.header.stamp.sec = msg.header.stamp.sec
            out.header.stamp.nanosec = msg.header.stamp.nsec
        else:
            out.header.stamp = self.get_clock().now().to_msg()

        out.dvl_type = GZ_TO_ROS_DVL_TYPE.get(msg.type, Dvl.DVL_TYPE_PISTON)
        out.velocity_mode = GZ_TO_ROS_VELOCITY_MODE.get(msg.target.type, Dvl.DVL_MODE_BOTTOM)

        # gz-sensors DVL reports velocity in FLU (x-forward, y-left, z-up);
        # ArduPilot's body frame (and VISION_POSITION_DELTA) expects FRD
        # (x-forward, y-right, z-down), matching the real A50 driver's
        # marine_acoustic_msgs/Dvl output. Flip Y and Z to convert FLU->FRD.
        v = msg.velocity.mean
        vx, vy, vz = v.x, -v.y, -v.z
        out.velocity = Vector3(x=vx, y=vy, z=vz)
        out.velocity_covar = self._resolve_covar9(msg.velocity.covariance)
        out.course_gnd = math.atan2(vy, vx)
        out.speed_gnd = math.hypot(vx, vy)
        out.sound_speed = 1481.0

        good_beams = [b for b in msg.beams if b.locked]
        out.num_good_beams = len(good_beams)
        out.beam_velocities_valid = out.num_good_beams >= self.min_good_beams
        out.beam_ranges_valid = out.num_good_beams >= self.min_good_beams

        beam_unit_vec = [Vector3(x=0.0, y=0.0, z=1.0)] * 4
        rng = [UNKNOWN_COVAR] * 4
        range_covar = [float(UNKNOWN_COVAR)] * 4
        beam_quality = [0.0] * 4
        beam_velocity = [0.0] * 4
        beam_velocity_covar = [float(UNKNOWN_COVAR)] * 4

        for beam in msg.beams:
            idx = beam.id - 1
            if not 0 <= idx < 4:
                continue
            beam_unit_vec[idx] = beam_unit_vector(beam.id)
            beam_velocity[idx] = beam.velocity.mean.x
            if beam.velocity.covariance:
                beam_velocity_covar[idx] = beam.velocity.covariance[0]
            if beam.locked:
                rng[idx] = beam.range.mean
                range_covar[idx] = beam.range.variance
                beam_quality[idx] = 100.0

        out.beam_unit_vec = beam_unit_vec
        out.range = rng
        out.range_covar = range_covar
        out.beam_quality = beam_quality
        out.beam_velocity = beam_velocity
        out.beam_velocity_covar = beam_velocity_covar

        if msg.target.type == 1 and msg.target.range.mean > 0.0:  # DVL_TARGET_BOTTOM
            out.altitude = msg.target.range.mean
        elif good_beams:
            tilt = math.radians(BEAM_GEOMETRY_DEG[1]["tilt"])
            mean_range = sum(b.range.mean for b in good_beams) / len(good_beams)
            out.altitude = math.cos(tilt) * mean_range
        else:
            out.altitude = -1.0

        self.pub.publish(out)

    @staticmethod
    def _resolve_covar9(covariance):
        covar = list(covariance)
        if len(covar) == 9:
            return covar
        diagonal = [float(UNKNOWN_COVAR)] * 9
        return diagonal


def main(args=None):
    rclpy.init(args=args)
    node = SimDvlBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
