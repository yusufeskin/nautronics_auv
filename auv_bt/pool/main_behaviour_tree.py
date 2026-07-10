#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.executors import MultiThreadedExecutor

import py_trees
import py_trees_ros.trees
import py_trees.console as console

# Import individual missions
import gate.gate_behaviour_tree

def create_root() -> py_trees.behaviour.Behaviour:
    # 1. MAIN TREE STRUCTURE
    # SuccessOnOne ensures that when the mission sequence finishes, the whole tree succeeds 
    # instead of being blocked by publishers that run indefinitely.
    root = py_trees.composites.Parallel(
        name="Main BT Root",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )

    # MAIN MISSIONS SEQUENCE (Runs each mission sequentially)
    all_missions_seq = py_trees.composites.Sequence("All Missions Sequence", memory=True)

    # -------------------------------------------------------------------------
    # ADD MISSIONS
    # -------------------------------------------------------------------------

    # --- 1. Gate Mission ---
    gate_root = gate.gate_behaviour_tree.create_root()
    gate_root.name = "Gate Mission Tree"
    all_missions_seq.add_child(gate_root)

    # --- 2. Future Missions ---
    # Example:
    # torpedo_root = torpedo.torpedo_behaviour_tree.create_root()
    # torpedo_root.name = "Torpedo Mission Tree"
    # all_missions_seq.add_child(torpedo_root)

    # Assemble Main Tree
    root.add_child(all_missions_seq)

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
