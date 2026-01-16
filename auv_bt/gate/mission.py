#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import py_trees_ros.trees
import py_trees.console as console
import rclpy
import sys
import gate.behaviors.imu
from rclpy.qos import qos_profile_sensor_data
import common_behaviors.state
import py_trees_ros.service_clients
from auv_interfaces.srv import SetVehicleMode
def tutorial_create_root() -> py_trees.behaviour.Behaviour:
    """
    Create a basic tree and start a 'Topics2BB' work sequence that
    will become responsible for data gathering behaviours.

    Returns:
        the root of the tree
    """

    root = py_trees.composites.Parallel(
        name="Tutorial One",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(
            synchronise=False
        )
    )

    topics2bb = py_trees.composites.Parallel(
    name="Topics2BB",
    policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
)
    imu2bb = gate.behaviors.imu.ToBlackboard(
        name="Imu2BB",
        topic_name="/imu0",
        qos_profile=qos_profile_sensor_data
    )

    mode_request = SetVehicleMode.Request()
    mode_request.mode_name = "ALT_HOLD"

    switch_mode_node = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request
    )

    status2bb = common_behaviors.state.ToBlackboard(
        name="status2BB",
        topic_name="/vehicle/state",
        qos_profile=py_trees_ros.utilities.qos_profile_unlatched())

    root.add_child(topics2bb)
    root.add_child(switch_mode_node)
    topics2bb.add_child(imu2bb)
    topics2bb.add_child(status2bb)


    return root


def tutorial_main():
    """
    Entry point for the demo script.
    """
    rclpy.init(args=None)
    root = tutorial_create_root()
    tree = py_trees_ros.trees.BehaviourTree(
        root=root,
        unicode_tree_debug=True
    )
    try:
        tree.setup(node_name="mission_control_tree", timeout=15.0)
    except py_trees_ros.exceptions.TimedOutError as e:
        console.logerror(console.red + "failed to setup the tree, aborting [{}]".format(str(e)) + console.reset)
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)
    except KeyboardInterrupt:
        # not a warning, nor error, usually a user-initiated shutdown
        console.logerror("tree setup interrupted")
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)

    tree.tick_tock(period_ms=1000.0)

    try:
        rclpy.spin(tree.node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        tree.shutdown()
        rclpy.try_shutdown()
