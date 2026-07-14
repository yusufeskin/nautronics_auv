#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import operator
import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
import py_trees.timers
import py_trees
import py_trees_ros.trees
import py_trees.console as console
import py_trees_ros.service_clients
import py_trees_ros.action_clients

import gate.behaviours.arrange_depth_action
import gate.behaviours.object2bb
import gate.behaviours.depth
import common_behaviors.state 

from auv_interfaces.action import VisualServoing
from auv_interfaces.action import YawAndScan
from auv_interfaces.action import BlindPush
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

    main_mission_sequence = py_trees.composites.Sequence("Slalom Mission", memory=True)

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

# 3. SLALOM MISSION BRANCH

    # 3.1 ALT_HOLD AND ARRANGE DEPTH TO -1.3
    mode_request_althold = SetVehicleMode.Request()
    mode_request_althold.mode_name = "ALT_HOLD"
    switch_mode_althold = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold
    )

    arrange_depth_node = gate.behaviours.arrange_depth_action.ArrangeDepthAction(
        name="Arrange Depth",
        topic_odom="/baro_data",
        topic_cmd="/cmd_vel",  
        target_depth=-1.3,
        tolerance=0.1,   
        speed=0.2             
    )

    # 3.2 ROTATE TO 10
    goal_msg_rotate_10 = YawAndScan.Goal()
    goal_msg_rotate_10.target_angle_deg = 10.0
    goal_msg_rotate_10.angular_speed = 0.05
    goal_msg_rotate_10.is_absolute = True
    
    rotate_10 = py_trees_ros.action_clients.FromConstant(
        name="Turn to Absolute 10",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg_rotate_10
    )

    # 3.3 BLIND PUSH 3s
    blind_push_3s_1 = py_trees_ros.action_clients.FromConstant(
        name="Blind Push 3s (1)",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=3.0,
            speed=0.3
        )
    )

    # 3.4 ROTATE TO 15
    goal_msg_rotate_15 = YawAndScan.Goal()
    goal_msg_rotate_15.target_angle_deg = 15.0
    goal_msg_rotate_15.angular_speed = 0.05
    goal_msg_rotate_15.is_absolute = True
    
    rotate_15 = py_trees_ros.action_clients.FromConstant(
        name="Turn to Absolute 15",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg_rotate_15
    )

    # 3.5 BLIND PUSH 3s
    blind_push_3s_2 = py_trees_ros.action_clients.FromConstant(
        name="Blind Push 3s (2)",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=3.0,
            speed=0.3
        )
    )

    # 3.6 ROTATE TO 20
    goal_msg_rotate_20 = YawAndScan.Goal()
    goal_msg_rotate_20.target_angle_deg = 20.0
    goal_msg_rotate_20.angular_speed = 0.05
    goal_msg_rotate_20.is_absolute = True
    
    rotate_20 = py_trees_ros.action_clients.FromConstant(
        name="Turn to Absolute 20",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg_rotate_20
    )

    # 3.7 BLIND PUSH 3s
    blind_push_3s_3 = py_trees_ros.action_clients.FromConstant(
        name="Blind Push 3s (3)",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=3.0,
            speed=0.3
        )
    )

    # 3.8 ROTATE TO 25
    goal_msg_rotate_25 = YawAndScan.Goal()
    goal_msg_rotate_25.target_angle_deg = 25.0
    goal_msg_rotate_25.angular_speed = 0.05
    goal_msg_rotate_25.is_absolute = True
    
    rotate_25 = py_trees_ros.action_clients.FromConstant(
        name="Turn to Absolute 25",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg_rotate_25
    )

    # 3.9 BLIND PUSH 10s
    blind_push_10s = py_trees_ros.action_clients.FromConstant(
        name="Blind Push 10s",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=10.0,
            speed=0.3
        )
    )

    # 3.10 SWITCH TO MANUAL
    mode_request_manual = SetVehicleMode.Request()
    mode_request_manual.mode_name = "MANUAL"
    switch_mode_manual = py_trees_ros.service_clients.FromConstant(
        name="SwitchToManual",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_manual
    )

# 4. ASSEMBLE MAIN MISSION
    main_mission_sequence.add_children([
        switch_mode_althold,
        arrange_depth_node,
        rotate_10,
        blind_push_3s_1,
        rotate_15,
        blind_push_3s_2,
        rotate_20,
        blind_push_3s_3,
        rotate_25,
        blind_push_10s,
        switch_mode_manual
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
        py_trees.display.render_dot_tree(root, name="real_gate_gorev_agaci")
        print("Ağaç başarıyla 'real_gate_gorev_agaci.svg' olarak kaydedildi!")
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