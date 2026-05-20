#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from auv_control import visual_servoing_action
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64
from geometry_msgs.msg import Point 

import py_trees
import py_trees_ros.trees
import py_trees.console as console
import py_trees_ros.service_clients
import py_trees_ros.action_clients

import behaviours.set_depth_action
import behaviours.set_attitude_action
import behaviours.depth
import behaviours.state 
import behaviours.attitude
from auv_interfaces.srv import SetVehicleMode
from auv_interfaces.action import VisualServoing


def create_publishers_branch() -> py_trees.composites.Parallel:
    publishers_parallel = py_trees.composites.Parallel(
        name="Publishers",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    depth2bb = py_trees_ros.subscribers.ToBlackboard(
        name="Depth2BB",
        topic_name="/baro_data2",
        topic_type=Float64,
        blackboard_variables={'depth': 'data'},
        qos_profile=qos_profile_sensor_data
    )

    mode2bb = behaviours.state.ToBlackboard(
        name="Mode2BB",
        topic_name="/vehicle/state",
        qos_profile=qos_profile_sensor_data
    )

    publishers_parallel.add_children([depth2bb, mode2bb])
    
    return publishers_parallel


def create_depth_sequence() -> py_trees.composites.Sequence:
    """Mod değiştirme ve hedeflenen derinliğe inme dizisini oluşturur."""
    set_depth_sequence = py_trees.composites.Sequence("Set Depth Sequence", memory=True)   

    mode_request_althold = SetVehicleMode.Request()
    mode_request_althold.mode_name = "ALT_HOLD"
    switch_mode_althold = py_trees_ros.service_clients.FromConstant(
        name="SwitchToAltHold",
        service_type=SetVehicleMode,
        service_name="/change_mode",
        service_request=mode_request_althold
    )

    depth_parallel = py_trees.composites.Parallel(
        name="Depth Control Parallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne() 
    )

    depth_action = behaviours.set_depth_action.SetDepthAction(
        name="Send Depth Target",
        target_depth=-1.5
    )

    depth_checker = behaviours.depth.DepthCheckerCondition(
        name="Depth Checker",
        topic="/baro_data2",
        target_depth=-1.5,
        tolerance=0.15 
    )

    depth_parallel.add_children([depth_action, depth_checker])
    set_depth_sequence.add_children([switch_mode_althold, depth_parallel])

    return set_depth_sequence



def create_search_and_align_sequence() -> py_trees.behaviour.Behaviour:
    """15 derece döner, arar ve bulursa hizalanır. Kaybederse Action'ı iptal edip başa sarar."""
    
    search_and_align_sequence = py_trees.composites.Sequence(
        name="Search and Align Sequence", 
        memory=True
    )

    turn_yaw_parallel = py_trees.composites.Parallel(
        name="Yaw Control Parallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )

    yaw_action = behaviours.set_attitude_action.SetAttitudeAction(
        name="Turn 15 Degrees",
        topic="/target_attitude",
        target_yaw=15.0,
        target_roll=0.0,
        target_pitch=0.0
    )

    yaw_checker = behaviours.attitude.AttitudeCheckerCondition(
        name="Check 15 Degrees",
        topic="/current_attitude",
        target_yaw=15.0,
        tolerance=2.0
    )

    turn_yaw_parallel.add_children([yaw_action, yaw_checker])

    target_points = [
        Point(x=254.0, y=23.0, z=0.0),
        Point(x=327.0, y=22.0, z=0.0),
        Point(x=327.0, y=96.0, z=0.0),
        Point(x=254.0, y=96.0, z=0.0)
    ]

    align_node = py_trees_ros.action_clients.FromConstant(
        name="Visual Servoing to Gate",
        action_type=VisualServoing,
        action_name="visual_servoing", 
        action_goal=VisualServoing.Goal(
            target_object="gate",
            target_points=target_points
        )
    )

    search_and_align_sequence.add_children([
        turn_yaw_parallel, 
        align_node,
    ])  

    loop = py_trees.decorators.Retry(
        name="Search Loop",
        child=search_and_align_sequence,
        num_retries=24
    )

    return loop  

def create_root() -> py_trees.behaviour.Behaviour:
    """Ana ağaç kökünü (root) oluşturur ve tüm alt dalları birleştirir."""
    # Alt ağaç parçalarını oluştur
    publishers_parallel = create_publishers_branch()
    set_depth_sequence = create_depth_sequence()
    loop = create_search_and_align_sequence()

    # Görev dizisini oluştur
    main_mission_sequence = py_trees.composites.Sequence("Main Mission Sequence", memory=True)
    main_mission_sequence.add_children([
        set_depth_sequence, 
        loop
    ])

    # Görevin sadece bir kez başarılı olana kadar çalışmasını sağla
    one_shot_main_mission = py_trees.decorators.OneShot(
        name="Mission OneShot",
        child=main_mission_sequence,
        policy=py_trees.common.OneShotPolicy.ON_SUCCESSFUL_COMPLETION
    )

    # Ana Kök: Yayıncılar (Sürekli) ve Görevler (Tek seferlik) paralel çalışır
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

    print("Starting Behavior Tree... (Press CTRL+C to stop)")
    
    try:
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