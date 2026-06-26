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

import common_behaviors.state 

from behaviours.attitude_to_bb import AttitudeToBlackboard
import behaviours.set_attitude_action
import behaviours.attitude

from auv_interfaces.action import BlindPush
from auv_interfaces.srv import SetVehicleMode
from auv_interfaces.action import YawAndScan



def create_root() -> py_trees.behaviour.Behaviour:



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

    publishers_parallel.add_children([mode2bb, attitude2bb])






    mode_request_althold = SetVehicleMode.Request()
    mode_request_althold.mode_name = "ALT_HOLD"
    switch_mode_althold = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold
    )







    goal_msg_90 = YawAndScan.Goal()
    goal_msg_90.target_angle_deg = 90.0
    goal_msg_90.angular_speed = 0.3

    yaw_turn_node1 = py_trees_ros.action_clients.FromConstant(
        name="Yaw Turn +90",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg_90
    )

    blind_push = py_trees_ros.action_clients.FromConstant(
        name="Blind Push Through Gate",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=7.0,
            speed=0.3
        )
    )

    yaw_turn_node2 = py_trees_ros.action_clients.FromConstant(
        name="Yaw Turn +90",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg_90
    )

    main_mission_sequence.add_children([
        switch_mode_althold,
        yaw_turn_node1,
        blind_push,
        yaw_turn_node2,
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