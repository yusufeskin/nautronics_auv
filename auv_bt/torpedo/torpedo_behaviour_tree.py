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
import behaviours.stop_vehicle
from std_srvs.srv import Trigger

from auv_interfaces.action import BlindPush, VisualServoing, YawAndScan
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
        target_depth=-1.5,
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

# 4. SEARCH TORPEDO (CHECK DETECTED) BRANCH

    check_detected_selector = py_trees.composites.Selector("Check if Detected", memory=True)

    check_torpedo_first = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Is Torpedo Detected?",
        check=py_trees.common.ComparisonExpression(
            variable="is_torpedo_found",
            value=True,
            operator=operator.eq)
    )
    
    search_torpedo_sequence = py_trees.composites.Sequence("Turn and Find Torpedo", memory=True)

    goal_msg = YawAndScan.Goal()
    goal_msg.target_angle_deg = 15.0
    goal_msg.angular_speed = 0.3  
    
    rotate_15_deg = py_trees_ros.action_clients.FromConstant(
        name="Turn 15 degrees",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg
    )

    check_torpedo_second = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Is Torpedo Detected?",
        check=py_trees.common.ComparisonExpression(
            variable="is_torpedo_found",
            value=True,
            operator=operator.eq)
    )

    search_torpedo_sequence.add_children([rotate_15_deg, check_torpedo_second])

    retry_search_torpedo = py_trees.decorators.Retry(
        name="retry (max)x24",
        child=search_torpedo_sequence,
        num_failures=24
    )

    check_detected_selector.add_children([check_torpedo_first, retry_search_torpedo])

# 5. ALIGN TO TORPEDO BRANCH

    allign_sequence = py_trees.composites.Sequence("Align to Torpedo", memory=True)

    mode_request_manual_2 = SetVehicleMode.Request()
    mode_request_manual_2.mode_name = "MANUAL"
    switch_mode_manual_second = py_trees_ros.service_clients.FromConstant(
        name="SwitchToManual",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_manual_2
    )

    target_points1 = [
        Point(x=245.0, y=288.0, z=0.0),  # Top Left
        Point(x=406.0, y=288.0, z=0.0),  # Top Right
        Point(x=406.0, y=449.0, z=0.0),  # Bottom Right
        Point(x=245.0, y=449.0, z=0.0)   # Bottom Left
    ]
    
    allign_node1 = py_trees_ros.action_clients.FromConstant(
        name="Visual Servoing to Torpedo",
        action_type=VisualServoing,
        action_name="/visual_servoing",
        action_goal=VisualServoing.Goal(
            target_object="torpedo",
            target_points=target_points1
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

    blind_push_node = py_trees_ros.action_clients.FromConstant(
        name="Blind Push Through Torpedo",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=2.0,
            speed=0.2
        )
    )

    stop_vehicle = behaviours.stop_vehicle.StopVehicle(name="Stop Vehicle", duration=3.0)


    fire_request_1 = Trigger.Request()
    fire_torpedo_node1 = py_trees_ros.service_clients.FromConstant(
        name="Fire Torpedo 1",
        service_type=Trigger,
        service_name="/torpedo/fire_service",
        service_request=fire_request_1
    )

    wait_5_sec = py_trees.timers.Timer("Wait 5s", duration=5.0)

    fire_request_2 = Trigger.Request()
    fire_torpedo_node2 = py_trees_ros.service_clients.FromConstant(
        name="Fire Torpedo 2",
        service_type=Trigger,
        service_name="/torpedo/fire_service",
        service_request=fire_request_2
    )

    allign_sequence.add_children([
        switch_mode_manual_second, 
        allign_node1,
        switch_mode_althold_second,
        blind_push_node,
        stop_vehicle,
        fire_torpedo_node1,
        wait_5_sec,
        fire_torpedo_node2
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