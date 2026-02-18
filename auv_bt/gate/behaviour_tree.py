#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import operator
import rclpy
import py_trees
import py_trees_ros.trees
import py_trees.console as console
import py_trees_ros.service_clients
from rclpy.qos import qos_profile_sensor_data
from auv_interfaces.srv import SetVehicleMode
import gate.behaviours.check_depth
import gate.behaviours.arrange_depth_action
import gate.behaviours.depth
import common_behaviours.state 
from auv_control import visual_servoing_action
from auv_control import blind_push_action
def create_root() -> py_trees.behaviour.Behaviour:
    root = py_trees.composites.Parallel(
        name="Main Parallel Root",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    publishers_sequence = py_trees.composites.Sequence("Publishers", memory=False)

    main_mission_sequence = py_trees.composites.Sequence("Stabilize and Hold", memory=True)

    depth2bb = gate.behaviours.depth.ToBlackboard(
        name="Depth2BB",
        topic_name="/odom",
        qos_profile=qos_profile_sensor_data
    )

    mode2bb = common_behaviours.state.ToBlackboard(
        name="Mode2BB",
        topic_name="/vehicle/state_mode"
    )

    check_arrange_depth_selector = py_trees.composites.Selector("Check Depth or Arrange", memory=False)

    check_depth_switch_althold_sequence = py_trees.composites.Sequence("Check Depth and Switch to AltHold", memory=False)

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

    switch_althold_sequence = py_trees.composites.Sequence("Switch to AltHold", memory=False)

    mode_request_althold_second = SetVehicleMode.Request()
    mode_request_althold_second.mode_name = "ALT_HOLD"
    switch_mode_althold_second = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold_second
    )

    check_althold_mode = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Check AltHold Mode",
        check=py_trees.common.ComparisonExpression(
            variable="vehicle_mode",
            value="ALT_HOLD",
            operator=operator.eq
        )
    )

    check_detected_selector = py_trees.composites.Selector("Check if Detected", memory=False)

    allign_sequence = py_trees.composites.Sequence("Align to Gate", memory=False)

    mode_request_manual = SetVehicleMode.Request()
    mode_request_manual.mode_name = "MANUAL"
    switch_mode_manual_second = py_trees_ros.service_clients.FromConstant(
        name="SwitchToManual",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_manual
    )

    check_manual_mode = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Check Manual Mode",
        check=py_trees.common.ComparisonExpression(
            variable="vehicle_mode",
            value="MANUAL",
            operator=operator.eq
        )
    )

    target_points = [
    (200.0, 100.0), # Top Left
    (440.0, 100.0), # Top Right
    (440.0, 380.0), # Bottom Right
    (200.0, 380.0)  # Bottom Left
]
    
    allign_node = visual_servoing_action.VisualServoingActionServer(
        name="Visual Servoing to Gate",
        target_object="gate",
        target_points=target_points  
    )

    blind_push_node = blind_push_action.BlindPushActionServer(
        name="Blind Push Through Gate",
        topic_cmd="/cmd_vel",
        duration=5.0,
        speed=0.3
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


    root.add_child(publishers_sequence)
    root.add_child(main_mission_sequence)
    
    publishers_sequence.add_children([depth2bb, mode2bb])
    main_mission_sequence.add_children([check_arrange_depth_selector])

    check_arrange_depth_selector.add_children([check_depth_switch_althold_sequence, arrange_depth_sequence])
    check_depth_switch_althold_sequence.add_children([check_depth, switch_mode_althold_first])
    arrange_depth_sequence.add_children([switch_mode_manual_first, arrange_depth_node, switch_althold_sequence])
    switch_althold_sequence.add_children([switch_mode_althold_second, check_althold_mode])
    
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
        rclpy.spin(tree.node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        tree.shutdown()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()