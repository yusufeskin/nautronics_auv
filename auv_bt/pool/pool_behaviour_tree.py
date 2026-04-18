#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

import py_trees
import py_trees_ros.trees
import py_trees.console as console
import py_trees_ros.service_clients
import py_trees_ros.action_clients

# Kendi yazdığın behaviour'lar
import behaviours.arrange_depth_action
import behaviours.depth
import behaviours.state 

# AUV Interface'leri
from auv_interfaces.action import BlindPush, YawAndScan
from auv_interfaces.srv import SetVehicleMode


def create_root() -> py_trees.behaviour.Behaviour:

    # ==========================================
    # 1. PUBLISHERS BRANCH (Sensör / Durum Yayıncıları)
    # ==========================================
    publishers_parallel = py_trees.composites.Parallel(
        name="Publishers",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    depth2bb = behaviours.depth.ToBlackboard(
        name="Depth2BB",
        topic_name="/baro_data",
        qos_profile=qos_profile_sensor_data
    )

    mode2bb = behaviours.state.ToBlackboard(
        name="Mode2BB",
        topic_name="/vehicle/state",
        qos_profile=qos_profile_sensor_data
    )

    publishers_parallel.add_children([depth2bb, mode2bb])

    # ==========================================
    # 2. ARRANGE DEPTH SEQUENCE (Mod ve Derinlik)
    # ==========================================
    arrange_depth_sequence = py_trees.composites.Sequence("Arrange Depth Sequence", memory=True)   

    mode_request_sta_1 = SetVehicleMode.Request()
    mode_request_sta_1.mode_name = "STABILIZE"
    switch_mode_stabilize_first = py_trees_ros.service_clients.FromConstant(
        name="SwitchToStabilize_1",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_sta_1
    )

    arrange_depth_node = behaviours.arrange_depth_action.ArrangeDepthAction(
        name="Arrange Depth Action",
        topic="/baro_data",
        topic_cmd="/cmd_vel",
        target_depth=-1.2,
        tolerance=0.1,   
        speed=0.15             
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
        switch_mode_stabilize_first, 
        arrange_depth_node, 
        switch_mode_althold_first
    ])

    # ==========================================
    # 3. SQUARE PATH SEQUENCE (Kare Çizme)
    # ==========================================
    square_path_sequence = py_trees.composites.Sequence("Square Path Sequence", memory=True)

    goal_msg_90 = YawAndScan.Goal()
    goal_msg_90.target_angle_deg = 90.0
    goal_msg_90.angular_speed = 0.3  

    mode_request_sta_2 = SetVehicleMode.Request()
    mode_request_sta_2.mode_name = "STABILIZE"
    switch_mode_stabilize_second = py_trees_ros.service_clients.FromConstant(
        name="SwitchToStabilize_2",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_sta_2
    )

    square_path_sequence.add_children([
        py_trees_ros.action_clients.FromConstant("Push (1)", BlindPush, "/blind_push", BlindPush.Goal(duration=3.0, speed=0.5)),
        py_trees_ros.action_clients.FromConstant("Turn 90 (1)", YawAndScan, "/yaw_and_scan", goal_msg_90),
        py_trees_ros.action_clients.FromConstant("Push (2)", BlindPush, "/blind_push", BlindPush.Goal(duration=3.0, speed=0.5)),
        py_trees_ros.action_clients.FromConstant("Turn 90 (2)", YawAndScan, "/yaw_and_scan", goal_msg_90),
        py_trees_ros.action_clients.FromConstant("Push (3)", BlindPush, "/blind_push", BlindPush.Goal(duration=3.0, speed=0.5)),
        py_trees_ros.action_clients.FromConstant("Turn 90 (3)", YawAndScan, "/yaw_and_scan", goal_msg_90),
        py_trees_ros.action_clients.FromConstant("Push (4)", BlindPush, "/blind_push", BlindPush.Goal(duration=3.0, speed=0.5)),
        py_trees_ros.action_clients.FromConstant("Turn 90 (4)", YawAndScan, "/yaw_and_scan", goal_msg_90),
        switch_mode_stabilize_second
    ])

    # ==========================================
    # 5. MAIN MISSION (Tüm Görevlerin Birleşimi)
    # ==========================================
    # İstenilen sıra: Derinlik -> Kare -> Kutlama
    main_mission_sequence = py_trees.composites.Sequence("Main Mission Sequence", memory=True)
    main_mission_sequence.add_children([
        arrange_depth_sequence,
        square_path_sequence
    ])

    # Tüm görevin sadece 1 kez çalışması için OneShot içine alıyoruz
    one_shot_main_mission = py_trees.decorators.OneShot(
        name="Mission OneShot",
        child=main_mission_sequence,
        policy=py_trees.common.OneShotPolicy.ON_SUCCESSFUL_COMPLETION
    )

    # ==========================================
    # 6. AĞACI BİRLEŞTİRME (TREE ASSEMBLY)
    # ==========================================
    # Kök düğüm (Root): Publisher'lar ve Ana Görev aynı anda (Parallel) çalışır
    root = py_trees.composites.Parallel(
        name="Main Parallel Root",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    root.add_children([publishers_parallel, one_shot_main_mission])

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

    print("Starting Behavior Tree... (Press CTRL+Z to stop)")
    
    try:
        # Tick tock arka planda çalışsın, executor ROS 2 callback'lerini yönetsin
        tree.tick_tock(period_ms=100.0)
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