#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import operator
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

import py_trees
import py_trees_ros.trees
import py_trees.console as console
import py_trees_ros.service_clients
import common_behaviors.state
import behaviours.attitude_to_bb
import behaviours.local_position_to_bb
import behaviours.relative_motion
import behaviours.wait_for_depth

from auv_interfaces.srv import SetVehicleMode
from std_srvs.srv import SetBool

TARGET_DEPTH = 0.35
LEG_DISTANCE1 = 2.0
LEG_DISTANCE2 = 2.3
TURN_DEG = 90.0
TARGET_SPEED_MPS = 0.2  # yavas/hassas hareket icin GUIDED hiz limiti
DEPTH_ARRIVAL_TOLERANCE = 0.2  # metre


def create_root() -> py_trees.behaviour.Behaviour:

    root = py_trees.composites.Parallel(
        name="Relative Zigzag Root",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    publishers_parallel = py_trees.composites.Parallel(
        name="Publishers",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    main_mission_sequence = py_trees.composites.Sequence("Relative Zigzag Mission", memory=True)

    one_shot_mission = py_trees.decorators.OneShot(
        name="Mission OneShot",
        child=main_mission_sequence,
        policy=py_trees.common.OneShotPolicy.ON_SUCCESSFUL_COMPLETION
    )

    mode2bb = common_behaviors.state.ToBlackboard(
        name="Mode2BB",
        topic_name="/vehicle/state",
        qos_profile=qos_profile_sensor_data
    )

    attitude2bb = behaviours.attitude_to_bb.AttitudeToBlackboard(
        name="Attitude2BB",
        topic_name="/current_attitude",
        qos_profile=qos_profile_sensor_data
    )

    local_position2bb = behaviours.local_position_to_bb.LocalPositionToBlackboard(
        name="LocalPosition2BB",
        topic_name="/vehicle/local_position",
        qos_profile=qos_profile_sensor_data
    )

    publishers_parallel.add_children([mode2bb, attitude2bb, local_position2bb])

    mode_request_guided = SetVehicleMode.Request()
    mode_request_guided.mode_name = "GUIDED"
    switch_to_guided = py_trees_ros.service_clients.FromConstant(
        name="SwitchToGuided",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_guided
    )

    wait_mode_guided = py_trees.decorators.Timeout(
        name="Timeout Wait GUIDED",
        duration=5.0,
        child=py_trees.behaviours.WaitForBlackboardVariableValue(
            name="Wait Mode GUIDED",
            check=py_trees.common.ComparisonExpression(
                variable="vehicle_mode",
                value="GUIDED",
                operator=operator.eq)
        )
    )

    retry_switch_guided = py_trees.composites.Sequence(
        name="Seq Switch Guided",
        memory=True,
        children=[switch_to_guided, wait_mode_guided]
    )
    retry_switch_guided = py_trees.decorators.Retry(
        name="Retry Switch Guided",
        child=retry_switch_guided,
        num_failures=10
    )

    arm_request = SetBool.Request()
    arm_request.data = True
    arm_vehicle = py_trees_ros.service_clients.FromConstant(
        name="ArmVehicle",
        service_type=SetBool,
        service_name="/arm",
        service_request=arm_request
    )

    set_speed = behaviours.relative_motion.SetSpeedAction(
        name="SetSlowSpeed",
        speed_mps=TARGET_SPEED_MPS
    )

    dive_set_target = behaviours.relative_motion.SetForwardTargetAction(
        name="Send Dive Target",
        distance=0.0,
        depth=TARGET_DEPTH
    )
    dive_wait_depth = behaviours.wait_for_depth.WaitForDepth(
        name="WaitForDiveDepth",
        target_depth=TARGET_DEPTH,
        tolerance=DEPTH_ARRIVAL_TOLERANCE
    )
    dive_step = py_trees.composites.Sequence(
        name="DiveInPlace",
        memory=True,
        children=[dive_set_target, dive_wait_depth]
    )
    forward1 = behaviours.relative_motion.create_forward_step(
        LEG_DISTANCE1, TARGET_DEPTH, "Forward1"
    )
    turn_right = behaviours.relative_motion.create_turn_step(
        +TURN_DEG, TARGET_DEPTH, "TurnRight"
    )
    forward2 = behaviours.relative_motion.create_forward_step(
        LEG_DISTANCE2, TARGET_DEPTH, "Forward2"
    )
    turn_left = behaviours.relative_motion.create_turn_step(
        -TURN_DEG, TARGET_DEPTH, "TurnLeft"
    )
    forward3 = behaviours.relative_motion.create_forward_step(
        LEG_DISTANCE1, TARGET_DEPTH, "Forward3"
    )

    mode_request_poshold = SetVehicleMode.Request()
    mode_request_poshold.mode_name = "POSHOLD"
    switch_to_poshold = py_trees_ros.service_clients.FromConstant(
        name="SwitchToPosHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_poshold
    )

    wait_mode_poshold = py_trees.decorators.Timeout(
        name="Timeout Wait POSHOLD",
        duration=5.0,
        child=py_trees.behaviours.WaitForBlackboardVariableValue(
            name="Wait Mode POSHOLD",
            check=py_trees.common.ComparisonExpression(
                variable="vehicle_mode",
                value="POSHOLD",
                operator=operator.eq)
        )
    )

    retry_switch_poshold = py_trees.composites.Sequence(
        name="Seq Switch PosHold",
        memory=True,
        children=[switch_to_poshold, wait_mode_poshold]
    )
    retry_switch_poshold = py_trees.decorators.Retry(
        name="Retry Switch PosHold",
        child=retry_switch_poshold,
        num_failures=10
    )

    main_mission_sequence.add_children([
        retry_switch_guided,
        arm_vehicle,
        set_speed,
        dive_step,
        forward1,
        turn_right,
        forward2,
        turn_left,
        forward3,
        retry_switch_poshold,
    ])

    root.add_child(publishers_parallel)
    root.add_child(one_shot_mission)

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

    print("Relative Zigzag Test BT Started... (Press CTRL+Z to stop)")
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
