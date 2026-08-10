#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import py_trees_ros.trees
import py_trees_ros.service_clients
import py_trees.console as console
import rclpy
import sys
from .set_parameter import SetYoloParameters
from .change_lifecycle import ChangeLifecycleState
from lifecycle_msgs.msg import Transition

def tutorial_create_root() -> py_trees.behaviour.Behaviour:
    root = py_trees.composites.Sequence(
        name="Setup YOLO Node",
        memory=True
    )

    yolo_params = {
        'model_name': 'realtorpedo_v2.pt', 
        'model_type': 'keypoint',
        'ema_alpha': 0.55,
        'distance_gate_threshold': 40.0,
        'miss_frames_limit': 15
    }

    set_all_params = SetYoloParameters(
        name="Set YOLO Settings",
        node_name="/universal_yolo_node",
        parameters_dict=yolo_params
    )

    configure_yolo = ChangeLifecycleState(
        name="Configure YOLO",
        node_name="/universal_yolo_node",
        transition_id=Transition.TRANSITION_CONFIGURE
    )

    activate_yolo = ChangeLifecycleState(
        name="Activate YOLO",
        node_name="/universal_yolo_node",
        transition_id=Transition.TRANSITION_ACTIVATE
    )
    
    idle = py_trees.behaviours.Running(name="Idle State")

    root.add_children([
        set_all_params,
        configure_yolo,
        activate_yolo,
        idle
    ])

    return root


def main():
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
        tree.setup(node_name="yolo_setup_manager_node", timeout=15.0)
    except py_trees_ros.exceptions.TimedOutError as e:
        console.logerror(console.red + f"failed to setup the tree, aborting [{str(e)}]" + console.reset)
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)
    except KeyboardInterrupt:
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

if __name__ == '__main__':
    main()