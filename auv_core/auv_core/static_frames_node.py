"""Publishes the vehicle's static frame tree from config/vehicle.yaml.

Sensor extrinsics exist in two places in most student stacks - the URDF used
by the simulator, and a pile of hard-coded offsets in whatever node happened
to need them - and they drift apart the first time a sensor is remounted.
Here the YAML is authoritative and this node is the only publisher of
base_link -> sensor transforms, so a remount is a one-line change.

Note this publishes only the rigid vehicle tree. ``odom -> base_link`` is the
estimator's output and ``map -> odom`` is the landmark corrector's, because
both are estimates rather than geometry.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

from auv_core.vehicle_config import default_config_path, load_vehicle_config


class StaticFramesNode(Node):
    def __init__(self):
        super().__init__("static_frames")

        self.declare_parameter("vehicle_config", default_config_path())
        path = self.get_parameter("vehicle_config").value

        config = load_vehicle_config(path)
        base = config.frames["base_link"]

        self._broadcaster = StaticTransformBroadcaster(self)
        stamp = self.get_clock().now().to_msg()

        transforms = []
        for name, mount in sorted(config.sensors.items()):
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = base
            tf.child_frame_id = mount.frame
            tf.transform.translation.x = float(mount.xyz[0])
            tf.transform.translation.y = float(mount.xyz[1])
            tf.transform.translation.z = float(mount.xyz[2])
            quat = mount.quat
            tf.transform.rotation.x = float(quat[0])
            tf.transform.rotation.y = float(quat[1])
            tf.transform.rotation.z = float(quat[2])
            tf.transform.rotation.w = float(quat[3])
            transforms.append(tf)
            self.get_logger().info(f"{base} -> {mount.frame} ({name})")

        self._broadcaster.sendTransform(transforms)
        self.get_logger().info(f"published {len(transforms)} static transforms from {path}")


def main(args=None):
    rclpy.init(args=args)
    node = StaticFramesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
