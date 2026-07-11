#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.executors import MultiThreadedExecutor

import py_trees
import py_trees.timers
import py_trees_ros.trees
import py_trees.console as console
from lifecycle_msgs.msg import Transition

# Import individual missions
import gate.gate_behaviour_tree
from lifecycle_manager.set_parameter import SetYoloParameters
from lifecycle_manager.change_lifecycle import ChangeLifecycleState

from std_srvs.srv import SetBool
from std_msgs.msg import UInt16MultiArray
import py_trees_ros.service_clients
import py_trees_ros.publishers
import py_trees_ros.utilities
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
    gate_mission_seq = py_trees.composites.Sequence("Gate Mission Seq", memory=True)

    arm_request = SetBool.Request()
    arm_request.data = True
    arm_node = py_trees_ros.service_clients.FromConstant(
        name="Arm Vehicle",
        service_type=SetBool,
        service_name="/arm",
        service_request=arm_request
    )

    throttle_msg = UInt16MultiArray()
    throttle_msg.data = [1500, 1500, 1500, 1500, 1600, 1500, 1500, 1500] 
    
    set_throttle_bb = py_trees.behaviours.SetBlackboardVariable(
        name="Set Throttle BB",
        variable_name="throttle_msg",
        variable_value=throttle_msg,
        overwrite=True
    )

    throttle_node = py_trees_ros.publishers.FromBlackboard(
        name="Apply Throttle",
        topic_name="/pwm_router",
        topic_type=UInt16MultiArray,
        qos_profile=py_trees_ros.utilities.qos_profile_latched(),
        blackboard_variable="throttle_msg"
    )

    timer_after_configure = py_trees.timers.Timer(name="Timer After Configure", duration=5.0)
    timer_after_activate = py_trees.timers.Timer(name="Timer After Activate", duration=5.0)

    gate_root = gate.gate_behaviour_tree.create_root()
    gate_root.name = "Gate Mission Tree"

    gate_mission_seq.add_children([
        arm_node,
        set_throttle_bb,
        throttle_node,
        configure_yolo, 
        timer_after_configure, 
        activate_yolo, 
        timer_after_activate, 
        gate_root
    ])
    root.add_child(gate_mission_seq)

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
