#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.executors import MultiThreadedExecutor

import py_trees
import py_trees_ros.trees
import py_trees.console as console
from lifecycle_msgs.msg import Transition

# Import individual missions
import gate.gate_behaviour_tree
from lifecycle_manager.set_parameter import SetYoloParameters
from lifecycle_manager.change_lifecycle import ChangeLifecycleState

def create_root() -> py_trees.behaviour.Behaviour:
    # 1. MAIN TREE STRUCTURE
    # We use a Sequence to run mission trees one by one
    root = py_trees.composites.Sequence("Main Missions Sequence", memory=True)

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


    # -------------------------------------------------------------------------
    # ADD MISSIONS
    # -------------------------------------------------------------------------

    # --- 1. Gate Mission ---
    gate_root = gate.gate_behaviour_tree.create_root()
    gate_root.name = "Gate Mission Tree"
    root.add_child(gate_root)

    # --- 2. Change YOLO Model for Torpedo ---
    # After Gate succeeds, we change the YOLO model dynamically
    yolo_params_torpedo = {
        'model_name': 'torpedo.pt'
    }
    set_yolo_torpedo = SetYoloParameters(
        name="Set YOLO Torpedo",
        node_name="/universal_yolo_node",
        parameters_dict=yolo_params_torpedo
    )
    root.add_child(set_yolo_torpedo)

    # --- 3. Future Missions (e.g. Torpedo) ---
    # torpedo_root = torpedo.torpedo_behaviour_tree.create_root()
    # torpedo_root.name = "Torpedo Mission Tree"
    # root.add_child(torpedo_root)

    return root

def main():
    rclpy.init(args=None)
    root = create_root()
    tree = py_trees_ros.trees.BehaviourTree(
        root=root,
        unicode_tree_debug=True
    )

    try:
        tree.setup(timeout=15)
        print(py_trees.display.unicode_tree(root))
    except py_trees_ros.exceptions.TimedOutError as e:
        console.logerror(console.red + "Setup Error: Connection failed [{}]".format(str(e)) + console.reset)
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)
    except KeyboardInterrupt:
        console.logerror("Initialization cancelled")
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)

    print("Starting Main Behavior Tree... (Press CTRL+Z to stop)")
    tree.tick_tock(period_ms=100.0)

    try:
        executor = MultiThreadedExecutor()
        executor.add_node(tree.node)
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        tree.shutdown()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
