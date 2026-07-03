#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import operator
import gate
import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

import py_trees
import py_trees_ros.trees
import py_trees.console as console
import py_trees_ros.service_clients
import py_trees_ros.action_clients
import common_behaviors.state 
import gate.behaviours.depth
import gate.behaviours.arrange_depth_action

from auv_interfaces.action import BlindPush
from auv_interfaces.action import YawAndScan
from auv_interfaces.srv import SetVehicleMode
from auv_interfaces.action import ReturnLoop


def create_root() -> py_trees.behaviour.Behaviour:

# 1. MAIN TREE STRUCTURE

    root = py_trees.composites.Parallel(
        name="Main Parallel Root",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    publishers_parallel = py_trees.composites.Parallel(
        name="Publishers",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    main_mission_sequence = py_trees.composites.Sequence("Stabilize and Hold", memory=True)

    one_shot_main_mission = py_trees.decorators.OneShot(
        name="Mission OneShot",
        child=main_mission_sequence,
        policy=py_trees.common.OneShotPolicy.ON_SUCCESSFUL_COMPLETION
    )

# 2. PUBLISHERS BRANCH

    mode2bb = common_behaviors.state.ToBlackboard(
        name="Mode2BB",
        topic_name="/vehicle/state",
        qos_profile=qos_profile_sensor_data
    )

    depth2bb = gate.behaviours.depth.ToBlackboard(
        name="Depth2BB",
        topic_name="/baro_data",
        qos_profile=qos_profile_sensor_data
    )

    publishers_parallel.add_children([mode2bb, depth2bb])

# 3. ARRANGE DEPTH BRANCH

    arrange_depth_sequence = py_trees.composites.Sequence("Arrange Depth", memory=True)

    mode_request_althold1 = SetVehicleMode.Request()
    mode_request_althold1.mode_name = "ALT_HOLD"
    switch_mode_althold1 = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold1
        )

    arrange_depth = gate.behaviours.arrange_depth_action.ArrangeDepthAction(
        name="Arrange Depth",
        topic_odom="/baro_data",
        topic_cmd="/cmd_vel",  
        target_depth=-0.5,
        tolerance=0.2,   
        speed=0.2             
    )

    blind_push1 = py_trees_ros.action_clients.FromConstant(
        name="Blind Push Through Gate",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=17.0,
            speed=0.2
        )
    )

    goal_msg = YawAndScan.Goal()
    goal_msg.target_angle_deg = 90.0
    goal_msg.angular_speed = 0.05 
    
    rotate_90_deg1 = py_trees_ros.action_clients.FromConstant(
        name="Turn 90 degrees",
        action_type=YawAndScan,
        action_name="/yaw_and_scan",
        action_goal=goal_msg
    )

    blind_push2 = py_trees_ros.action_clients.FromConstant(
        name="Blind Push Through Gate",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=17.0,
            speed=0.2
        )
    )

    return_loop = py_trees_ros.action_clients.FromConstant(
        name="Return Loop to Start",
        action_type=ReturnLoop,
        action_name="/return_loop",
        action_goal=ReturnLoop.Goal(
            duration=20.0,
            radius=3.0
        )
    )

    rotate_90_deg2 = py_trees_ros.action_clients.FromConstant(
        name="Turn 90 degrees",
        action_type=YawAndScan,
        action_name="/yaw_and_scan",
        action_goal=goal_msg
    )

    blind_push3 = py_trees_ros.action_clients.FromConstant(
        name="Blind Push Through Gate",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=17.0,
            speed=0.2
        )
    )

    rotate_90_deg3 = py_trees_ros.action_clients.FromConstant(
        name="Turn 90 degrees",
        action_type=YawAndScan,
        action_name="/yaw_and_scan",
        action_goal=goal_msg
    )

    blind_push4 = py_trees_ros.action_clients.FromConstant(
        name="Blind Push Through Gate",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=17.0,
            speed=0.2
        )
    )


    arrange_depth_sequence.add_children([
        switch_mode_althold1,
        arrange_depth,
        blind_push1,
        rotate_90_deg1,
        blind_push2,
        return_loop,
        rotate_90_deg2,
        blind_push3,
        rotate_90_deg3,
        blind_push4
    ])


    main_mission_sequence.add_children([
        arrange_depth_sequence,
    ])

    root.add_child(publishers_parallel)
    root.add_child(one_shot_main_mission)

    return root

# ROS 2 MAIN EXECUTION

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

    print("Starting Behavior Tree... (Press CTRL+Z to stop)")
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