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
import gate.behaviours.align_middle # Eklediğimiz ortalama kodu (CenterTarget)

from auv_interfaces.action import YawAndScan
from auv_interfaces.srv import SetVehicleMode
from auv_interfaces.action import BlindPush


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
        target_depth=-0.5,
        tolerance=0.1,   
        speed=0.2             
    )

    arrange_depth_sequence.add_children([
        switch_mode_althold1, 
        arrange_depth_node,
    ])

# 4. SEARCH AND PUSH BRANCH
    search_and_push_sequence = py_trees.composites.Sequence("Search, Align and Push Mission", memory=True)
    
    # 4.1 SEARCH SELECTOR
    find_target_selector = py_trees.composites.Selector("Find Target", memory=False)
    
    # NOT: check_gate_first kısmını aradığın hedefe göre değiştirebilirsin
    # yolo modelindeki ismine göre is_gate_found değil is_compass_found vs olabilir
    check_gate_first = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Is Target Detected?",
        check=py_trees.common.ComparisonExpression(
            variable="is_gate_found",
            value=True,
            operator=operator.eq)
    )
    
    search_gate_sequence = py_trees.composites.Sequence("Turn and Find Target", memory=True)

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
        name="Is Target Detected?",
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
    
    # 4.2 ALIGN / CENTER TARGET (YENİ EKLENEN KISIM)
    center_target_node = gate.behaviours.align_middle.CenterTarget(
        name="Ortala - Gate",
        target_class="gate", # Hedefin sınıf ismi (YOLO'daki adı)
        error_tol_x=40.0,
        error_tol_y=40.0,
        settle_time=2.0
    )

    # 4.3 BLIND PUSH
    blind_push_node = py_trees_ros.action_clients.FromConstant(
        name="Blind Push to Target",
        action_type=BlindPush,
        action_name="/blind_push",
        action_goal=BlindPush.Goal(
            duration=15.0,
            speed=0.3
        )
    )
    
    # Arama -> Ortalama -> Düz Gitme (Kör Sürüş)
    search_and_push_sequence.add_children([find_target_selector, center_target_node, blind_push_node])

# 5. ASSEMBLE MAIN MISSION

    main_mission_sequence.add_children([
        wait_40_secs,
        arrange_depth_sequence, 
        search_and_push_sequence
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
        py_trees.display.render_dot_tree(root, name="qualification2_tree")
        print("Ağaç başarıyla 'qualification2_tree.svg' olarak kaydedildi!")
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
