#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
import py_trees.timers
import py_trees
import py_trees_ros.trees
import py_trees.console as console
import py_trees_ros.service_clients
import py_trees_ros.action_clients

import gate.behaviours.arrange_depth_action
import gate.behaviours.depth
import common_behaviors.state 

from auv_interfaces.action import YawAndScan
from auv_interfaces.srv import SetVehicleMode
from auv_interfaces.action import BlindPush
from auv_interfaces.action import Roll

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

    depth2bb = gate.behaviours.depth.ToBlackboard(
        name="Depth2BB",
        topic_name="/baro_data",
        qos_profile=qos_profile_sensor_data
    )

    mode2bb = common_behaviors.state.ToBlackboard(
        name="Mode2BB",
        topic_name="/vehicle/state",
        qos_profile=qos_profile_sensor_data
    )

    publishers_parallel.add_children([depth2bb, mode2bb])

# 3. ARRANGE DEPTH BRANCH

    wait_40_secs = py_trees.timers.Timer(name="Wait 40 Seconds", duration=40.0)

    arrange_depth_sequence = py_trees.composites.Sequence("Arrange Depth", memory=True)   

    mode_request_althold1 = SetVehicleMode.Request()
    mode_request_althold1.mode_name = "ALT_HOLD"
    switch_mode_althold1 = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold1
    )

    arrange_depth_node = gate.behaviours.arrange_depth_action.ArrangeDepthAction(
        name="Arrange Depth",
        topic_odom="/baro_data",
        topic_cmd="/cmd_vel",  
        target_depth=-0.7,
        tolerance=0.1,   
        speed=0.2             
    )

    arrange_depth_sequence.add_children([
        switch_mode_althold1, 
        arrange_depth_node,
    ])

# 4. ROTATE AND PUSH BRANCH
    search_and_push_sequence = py_trees.composites.Sequence("Rotate and Push Mission", memory=True)
    
    # 4.1 ROTATE TO ABSOLUTE 0
    goal_msg_rotate = YawAndScan.Goal()
    goal_msg_rotate.target_angle_deg = 0.0
    goal_msg_rotate.angular_speed = 0.05
    goal_msg_rotate.is_absolute = True
    
    rotate_to_zero = py_trees_ros.action_clients.FromConstant(
        name="Turn to Absolute 0",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg_rotate
    )
    
    # 4.2 BLIND PUSH
    blind_push_node = py_trees_ros.action_clients.FromConstant(
        name="Blind Push to Gate",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=13.0,
            speed=0.3
        )
    )
    
    search_and_push_sequence.add_children([rotate_to_zero, blind_push_node])

    finish_roll_sequence = py_trees.composites.Sequence("Finish 360x2 Roll", memory=True)


    mode_request_acro = SetVehicleMode.Request()
    mode_request_acro.mode_name = "ACRO"
    switch_mode_acro1 = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAcro",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_acro
    )

    goal_msg_roll = Roll.Goal()
    goal_msg_roll.target_angle_deg = 360.0
    goal_msg_roll.angular_speed = 0.4

    roll_360_node1 = py_trees_ros.action_clients.FromConstant(
        name="Roll 360",
        action_type=Roll,
        action_name="/roll",
        action_goal=goal_msg_roll
    )

    mode_request_althold2 = SetVehicleMode.Request()
    mode_request_althold2.mode_name = "ALT_HOLD"
    switch_mode_althold2 = py_trees_ros.service_clients.FromConstant(
        name="SwitchBackToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold2
    )

    wait_2s_node = py_trees.timers.Timer(
        name=f"Wait 2s",
        duration=2.0
    )

    mode_request_acro = SetVehicleMode.Request()
    mode_request_acro.mode_name = "ACRO"
    switch_mode_acro2 = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAcro",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_acro
    )

    roll_360_node2 = py_trees_ros.action_clients.FromConstant(
        name="Roll 360",
        action_type=Roll,
        action_name="/roll",
        action_goal=goal_msg_roll
    )

    mode_request_althold3 = SetVehicleMode.Request()
    mode_request_althold3.mode_name = "ALT_HOLD"
    switch_mode_althold3 = py_trees_ros.service_clients.FromConstant(
        name="SwitchBackToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold3
    )

    finish_roll_sequence.add_children([switch_mode_acro1, 
                                       roll_360_node1, 
                                       switch_mode_althold2, 
                                       wait_2s_node, 
                                       switch_mode_acro2, 
                                       roll_360_node2, 
                                       switch_mode_althold3])

# 5. ASSEMBLE MAIN MISSION

    main_mission_sequence.add_children([
        wait_40_secs,
        arrange_depth_sequence, 
        search_and_push_sequence,
        finish_roll_sequence
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
        py_trees.display.render_dot_tree(root, name="hadi_oglum_tree")
        print("Ağaç başarıyla 'hadi_oglum_tree.svg' olarak kaydedildi!")
    except Exception as e:
        print(f"Ağaç çizilirken bir hata oluştu (Önemli değil, göreve devam edilecek): {e}")

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