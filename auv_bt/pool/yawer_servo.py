#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import operator
import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

import py_trees
import py_trees_ros.trees
import py_trees.console as console
import py_trees_ros.service_clients
import py_trees_ros.action_clients

import gate.behaviours.object2bb
import common_behaviors.state 
from behaviours.attitude_to_bb import AttitudeToBlackboard

from auv_interfaces.action import VisualServoing
from auv_interfaces.action import YawAndScan
from auv_interfaces.srv import SetVehicleMode


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

    main_mission_sequence = py_trees.composites.Sequence("Torpedo Search Mission", memory=True)

    one_shot_main_mission = py_trees.decorators.OneShot(
        name="Torpedo Search OneShot",
        child=main_mission_sequence,
        policy=py_trees.common.OneShotPolicy.ON_SUCCESSFUL_COMPLETION
    )

# 2. PUBLISHERS BRANCH

    mode2bb = common_behaviors.state.ToBlackboard(
        name="Mode2BB",
        topic_name="/vehicle/state",
        qos_profile=qos_profile_sensor_data
    )

    attitude2bb = AttitudeToBlackboard(
        name="Attitude2BB",
        topic_name="/current_attitude",
        qos_profile=qos_profile_sensor_data
    )

    object2bb = gate.behaviours.object2bb.ToBlackboard(
        name="Object2BB",
        topic_name="/yolo_detections",  
        qos_profile=qos_profile_sensor_data
    )

    publishers_parallel.add_children([mode2bb, object2bb, attitude2bb])

# 3. TORPEDO SEARCH SEQUENCE

    mode_request_althold_1 = SetVehicleMode.Request()
    mode_request_althold_1.mode_name = "ALT_HOLD"
    switch_mode_althold_1 = py_trees_ros.service_clients.FromConstant(
        name="Initial AltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold_1
    )

    search_step_sequence = py_trees.composites.Sequence("15 Degree Search Step", memory=False)

    goal_msg_15 = YawAndScan.Goal()
    goal_msg_15.target_angle_deg = 15.0
    goal_msg_15.angular_speed = 0.3

    yaw_turn_node = py_trees_ros.action_clients.FromConstant(
        name="Yaw Turn +15",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg_15
    )

    target_points = [
        Point(x=254.0, y=23.0, z=0.0),  # Top Left
        Point(x=327.0, y=22.0, z=0.0),  # Top Right
        Point(x=327.0, y=96.0, z=0.0),  # Bottom Right
        Point(x=254.0, y=96.0, z=0.0)   # Bottom Left
    ]
    
    allign_node = py_trees_ros.action_clients.FromConstant(
        name="Visual Servoing to Torpedo",
        action_type=VisualServoing,
        action_name="/visual_servoing",
        action_goal=VisualServoing.Goal(
            target_object="torpedo",
            target_points=target_points
        )
    )

    search_step_sequence.add_children([
        yaw_turn_node,
        allign_node 
    ])  

    search_loop = py_trees.decorators.Retry(
        name="360 Degree Search Loop",
        child=search_step_sequence,
        num_failures=24
    )

    main_mission_sequence.add_children([
        switch_mode_althold_1, 
        search_loop
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