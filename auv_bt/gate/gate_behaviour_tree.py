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

    object2bb = gate.behaviours.object2bb.ToBlackboard(
        name="Object2BB",
        topic_name="/object_3d_poses",  
        qos_profile=qos_profile_sensor_data
    )

    publishers_parallel.add_children([depth2bb, mode2bb, object2bb])

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

    arrange_depth_node = gate.behaviours.arrange_depth_action.ArrangeDepthAction(
        name="Arrange Depth",
        topic_odom="/baro_data",
        topic_cmd="/cmd_vel",  
        target_depth=-1.0,
        tolerance=0.1,   
        speed=0.2             
    )

    arrange_depth_sequence.add_children([
        switch_mode_althold1, 
        arrange_depth_node,
    ])

# 4. SEARCH AND SERVO (RECOVERY LOOP) BRANCH
    search_and_servo_loop = py_trees.composites.Sequence("Search and Servo Mission", memory=False)
    
    # 4.1 SEARCH SELECTOR
    find_target_selector = py_trees.composites.Selector("Find Gate", memory=False)
    
    check_gate_first = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Is Gate Detected?",
        check=py_trees.common.ComparisonExpression(
            variable="is_gate_found",
            value=True,
            operator=operator.eq)
    )
    
    search_gate_sequence = py_trees.composites.Sequence("Turn and Find Gate", memory=True)

    goal_msg_search = YawAndScan.Goal()
    goal_msg_search.target_angle_deg = 15.0
    goal_msg_search.angular_speed = 0.05
    
    rotate_15_deg = py_trees_ros.action_clients.FromConstant(
        name="Turn 15 degrees",
        action_type=YawAndScan,
        action_name="/yaw_and_scan", 
        action_goal=goal_msg_search
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
        name="Retry Search (max)x24",
        child=search_gate_sequence,
        num_failures=24
    )
    
    find_target_selector.add_children([check_gate_first, retry_search_gate])
    
    # 4.2 VISUAL SERVOING
    target_points = [
        Point(x=106.0, y=107.0, z=0.0),  # Top Left
        Point(x=360.0, y=107.0, z=0.0),  # Top Right
        Point(x=367.0, y=390.0, z=0.0),  # Bottom Right
        Point(x=98.0, y=381.0, z=0.0)   # Bottom Left
    ]

    
    visual_servo_node = py_trees_ros.action_clients.FromConstant(
        name="Visual Servoing to gate",
        action_type=VisualServoing,
        action_name="/visual_servoing",
        action_goal=VisualServoing.Goal(
            target_object="gate",
            target_points=target_points
        )
    )
    
    search_and_servo_loop.add_children([find_target_selector, visual_servo_node])
    
    # 4.3 WRAP IN RETRY LOOP 
    robust_servo_mission = py_trees.decorators.Retry(
        name="Robust Servo Recovery Loop",
        child=search_and_servo_loop,
        num_failures=100
    )


        # ------------------------------------------
    blind_push1 = py_trees_ros.action_clients.FromConstant(
        name="Blind Push Through Gate",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=10.0,
            speed=0.2
        )
    )


# 5. FINISH YAW BRANCH (90x8)

    finish_yaw_sequence = py_trees.composites.Sequence("Finish 90x8 Yaw", memory=True)
    
    for i in range(8):
        goal_msg_90 = YawAndScan.Goal()
        goal_msg_90.target_angle_deg = 90.0
        goal_msg_90.angular_speed = 0.1
        
        turn_90_node = py_trees_ros.action_clients.FromConstant(
            name=f"Turn 90 degrees ({i+1}/8)",
            action_type=YawAndScan,
            action_name="/yaw_and_scan", 
            action_goal=goal_msg_90
        )
        
        wait_1s_node = py_trees.timers.Timer(
            name=f"Wait 1s ({i+1}/8)",
            duration=1.0
        )

        
        finish_yaw_sequence.add_children([turn_90_node, wait_1s_node])


# 6. ASSEMBLE MAIN MISSION

    main_mission_sequence.add_children([
        arrange_depth_sequence, 
        robust_servo_mission, 
        blind_push1,
        finish_yaw_sequence
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