#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import operator
import rclpy
import py_trees
import py_trees_ros.trees
import py_trees.console as console
import py_trees_ros.service_clients
import py_trees_ros.action_clients
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
import gate.behaviours.check_depth
import gate.behaviours.arrange_depth_action
import gate.behaviours.object2bb
import gate.behaviours.depth
import common_behaviours.state 
from auv_interfaces.action import VisualServoing
from auv_interfaces.action import BlindPush
from auv_interfaces.action import YawAndScan
from auv_interfaces.srv import SetVehicleMode


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

    one_shot_mission = py_trees.decorators.OneShot(
        name="Mission OneShot",
        child=main_mission_sequence,
        policy=py_trees.common.OneShotPolicy.ON_SUCCESSFUL_COMPLETION)

    depth2bb = gate.behaviours.depth.ToBlackboard(
        name="Depth2BB",
        topic_name="/odom",
        qos_profile=qos_profile_sensor_data
    )

    mode2bb = common_behaviours.state.ToBlackboard(
        name="Mode2BB",
        topic_name="/vehicle/state_mode"
    )

    object2bb = gate.behaviours.object2bb.ToBlackboard(
        name="Object2BB",
        topic_name="/yolo_detections",  
        qos_profile=qos_profile_sensor_data
    )

    check_arrange_depth_selector = py_trees.composites.Selector("Check Depth or Arrange", memory=True)

    check_depth_switch_althold_sequence = py_trees.composites.Sequence("Check Depth and Switch to AltHold", memory=True)

    arrange_depth_sequence = py_trees.composites.Sequence("Arrange Depth", memory=False)   

    check_depth = gate.behaviours.check_depth.CheckDepth(
        name="Depth OK? (Main)",
        topic_name="/odom" 
    ) 

    mode_request_althold_first = SetVehicleMode.Request()
    mode_request_althold_first.mode_name = "ALT_HOLD"
    switch_mode_althold_first = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold_first
    )

    mode_request_manual = SetVehicleMode.Request()
    mode_request_manual.mode_name = "MANUAL"
    switch_mode_manual_first = py_trees_ros.service_clients.FromConstant(
        name="SwitchToManual",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_manual
    )

    arrange_depth_node = gate.behaviours.arrange_depth_action.ArrangeDepthAction(
        name="Arrange Depth",
        topic_odom="/odom",
        topic_cmd="/cmd_vel",  
        target_depth=-1.5,
        tolerance=0.2,   
        speed=0.2             
    )

    mode_request_althold_second = SetVehicleMode.Request()
    mode_request_althold_second.mode_name = "ALT_HOLD"
    switch_mode_althold_second = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold_second
    )

    check_detected_selector = py_trees.composites.Selector("Check if Detected", memory=True)

    check_gate_first = py_trees.behaviours.CheckBlackboardVariableValue(
    name="Is Gate Detected?",
    check=py_trees.common.ComparisonExpression(
        variable="is_gate_found",
        value=True,
        operator=operator.eq
    )
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
        operator=operator.eq
    )
)
    
    retry_search = py_trees.decorators.Retry(
    name="retry (max)x24",
    child=search_gate_sequence,
    num_failures=24
)

    allign_sequence = py_trees.composites.Sequence("Align to Gate", memory=False)

    mode_request_manual = SetVehicleMode.Request()
    mode_request_manual.mode_name = "MANUAL"
    switch_mode_manual_second = py_trees_ros.service_clients.FromConstant(
        name="SwitchToManual",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_manual
    )

    target_points = [
    (200.0, 100.0), # Top Left
    (440.0, 100.0), # Top Right
    (440.0, 380.0), # Bottom Right
    (200.0, 380.0)  # Bottom Left
]
    
    allign_node = py_trees_ros.action_clients.FromConstant(
        name="Visual Servoing to Gate",
        action_type=VisualServoing,
        action_name="/visual_servoing_action",
        action_goal=VisualServoing.Goal(
            target_object="gate",
            target_points=target_points
        )
    )

    blind_push_node = py_trees_ros.action_clients.FromConstant(
        name="Blind Push Through Gate",
        action_type=BlindPush,
        action_name="/blind_push_action",
        action_goal=BlindPush.Goal(
            duration=5.0,
            speed=0.3
        )
    )

    switch_althold_sequence_second = py_trees.composites.Sequence("Switch to AltHold", memory=False)

    mode_request_althold_third = SetVehicleMode.Request()
    mode_request_althold_third.mode_name = "ALT_HOLD"
    switch_mode_althold_third = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold_third
    )

    check_althold_mode_second = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Check AltHold Mode",
        check=py_trees.common.ComparisonExpression(
            variable="vehicle_mode",
            value="ALT_HOLD",
            operator=operator.eq
        )
    )


    root.add_child(publishers_parallel)
    root.add_child(one_shot_mission)
    
    publishers_parallel.add_children([depth2bb, mode2bb, object2bb])
    main_mission_sequence.add_children([check_arrange_depth_selector, check_detected_selector, allign_sequence])

    check_arrange_depth_selector.add_children([check_depth_switch_althold_sequence, arrange_depth_sequence])   
    arrange_depth_sequence.add_children([switch_mode_manual_first, arrange_depth_node, switch_mode_althold_second])
    check_depth_switch_althold_sequence.add_children([check_depth, switch_mode_althold_first])
    check_detected_selector.add_children([check_gate_first, retry_search])
    search_gate_sequence.add_children([rotate_15_deg, check_gate_second])
    allign_sequence.add_children([switch_mode_manual_second, allign_node, blind_push_node, switch_althold_sequence_second])
    switch_althold_sequence_second.add_children([switch_mode_althold_third, check_althold_mode_second])

    
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
    tree.tick_tock(period_ms=100.0)

    # --- DEĞİŞEN KISIM BURASI ---
    try:
        # Tek işlemci yerine Çoklu işlemci kullanıyoruz
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

if __name__ == '__main__':
    main()