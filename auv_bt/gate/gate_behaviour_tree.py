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

import behaviours.arrange_depth_action
import behaviours.object2bb
import behaviours.depth
import behaviours.state 

from auv_interfaces.action import VisualServoing
from auv_interfaces.action import BlindPush
from auv_interfaces.action import YawAndScan
from auv_interfaces.action import Roll  
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

    main_mission_sequence = py_trees.composites.Sequence("Stabilize and Hold", memory=True)

    one_shot_main_mission = py_trees.decorators.OneShot(
        name="Mission OneShot",
        child=main_mission_sequence,
        policy=py_trees.common.OneShotPolicy.ON_SUCCESSFUL_COMPLETION
    )

# 2. PUBLISHERS BRANCH

    depth2bb = behaviours.depth.ToBlackboard(
        name="Depth2BB",
        topic_name="/odom",
        qos_profile=qos_profile_sensor_data
    )

    mode2bb = behaviours.state.ToBlackboard(
        name="Mode2BB",
        topic_name="/vehicle/state",
        qos_profile=qos_profile_sensor_data
    )

    object2bb = behaviours.object2bb.ToBlackboard(
        name="Object2BB",
        topic_name="/yolo_detections",  
        qos_profile=qos_profile_sensor_data
    )

    publishers_parallel.add_children([depth2bb, mode2bb, object2bb])

# 3. ARRANGE DEPTH BRANCH

    arrange_depth_sequence = py_trees.composites.Sequence("Arrange Depth", memory=True)   

    mode_request_manual_1 = SetVehicleMode.Request()
    mode_request_manual_1.mode_name = "MANUAL"
    switch_mode_manual_first = py_trees_ros.service_clients.FromConstant(
        name="SwitchToManual",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_manual_1
    )

    arrange_depth_node = behaviours.arrange_depth_action.ArrangeDepthAction(
        name="Arrange Depth",
        topic_odom="/odom",
        topic_cmd="/cmd_vel",  
        target_depth=-0.5,
        tolerance=0.2,   
        speed=0.2             
    )

    mode_request_althold_first = SetVehicleMode.Request()
    mode_request_althold_first.mode_name = "ALT_HOLD"
    switch_mode_althold_first = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold_first
    )

    arrange_depth_sequence.add_children([
        switch_mode_manual_first, 
        arrange_depth_node, 
        switch_mode_althold_first
    ])

# 4. SEARCH GATE (CHECK DETECTED) BRANCH

    check_detected_selector = py_trees.composites.Selector("Check if Detected", memory=True)

    check_gate_first = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Is Gate Detected?",
        check=py_trees.common.ComparisonExpression(
            variable="is_gate_found",
            value=True,
            operator=operator.eq)
    )
    
    search_gate_sequence = py_trees.composites.Sequence("Turn and Find Gate", memory=True)

    goal_msg = YawAndScan.Goal()
    goal_msg.target_angle_deg = 15.0
    goal_msg.angular_speed = 0.3  
    
    rotate_15_deg = py_trees_ros.action_clients.FromConstant(
        name="Turn 15 degrees",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg
    )

    check_gate_second = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Is Gate Detected?",
        check=py_trees.common.ComparisonExpression(
            variable="is_gate_found",
            value=True,
            operator=operator.eq)
    )

    search_gate_sequence.add_children([rotate_15_deg, check_gate_second])

    retry_search_gate = py_trees.decorators.Retry(
        name="retry (max)x24",
        child=search_gate_sequence,
        num_failures=24
    )

    check_detected_selector.add_children([check_gate_first, retry_search_gate])

# 5. ALIGN TO GATE BRANCH

    allign_sequence = py_trees.composites.Sequence("Align to Gate", memory=True)

    mode_request_manual_2 = SetVehicleMode.Request()
    mode_request_manual_2.mode_name = "MANUAL"
    switch_mode_manual_second = py_trees_ros.service_clients.FromConstant(
        name="SwitchToManual",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_manual_2
    )

    target_points = [
        Point(x=254.0, y=24.0, z=0.0),  # Top Left
        Point(x=325.0, y=24.0, z=0.0),  # Top Right
        Point(x=325.0, y=94.0, z=0.0),  # Bottom Right
        Point(x=254.0, y=94.0, z=0.0)   # Bottom Left
    ]
    
    allign_node = py_trees_ros.action_clients.FromConstant(
        name="Visual Servoing to Gate",
        action_type=VisualServoing,
        action_name="/visual_servoing",
        action_goal=VisualServoing.Goal(
            target_object="gate",
            target_points=target_points
        )
    )

    mode_request_althold_second = SetVehicleMode.Request()
    mode_request_althold_second.mode_name = "ALT_HOLD"
    switch_mode_althold_second = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold_second
    )


    blind_push_node1 = py_trees_ros.action_clients.FromConstant(
        name="Blind Push Through Gate",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=4.0,
            speed=0.3
        )
    )

    roll_node = py_trees_ros.action_clients.FromConstant(
        name="720 Degree Roll",
        action_type=Roll,
        action_name="/roll",
        action_goal=Roll.Goal(
            target_angle_deg=720.0,
            angular_speed=0.5
        )
    )

    blind_push_node2 = py_trees_ros.action_clients.FromConstant(
        name="Blind Push Through Gate",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=2.0,
            speed=0.3
        )
    )

    allign_sequence.add_children([
        switch_mode_manual_second, 
        allign_node, 
        switch_mode_althold_second,
        blind_push_node1, 
        roll_node,
        blind_push_node2,
    ])

# 6. ASSEMBLE MAIN MISSION

    main_mission_sequence.add_children([
        arrange_depth_sequence, 
        check_detected_selector, 
        allign_sequence
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