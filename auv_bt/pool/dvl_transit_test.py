#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import operator
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

import py_trees
import py_trees_ros.trees
import py_trees.console as console
import py_trees_ros.service_clients
import py_trees.timers
import common_behaviors.state
import behaviours.wait_for_arrival

from auv_interfaces.srv import SetVehicleMode, GoToGpsTarget
from std_srvs.srv import SetBool

# ==========================================
# KOORDİNATLAR
# ==========================================
BASLANGIC_LAT = 39.85701944444445   # Havuz orijini (İskele)
BASLANGIC_LON = 32.69128611111111
HEDEF_LAT     = 39.856914           # 39°51'24.89"N
HEDEF_LON     = 32.691067           # 32°41'27.84"E
HEDEF_DERINLIK = 1.5                # metre (NED - aşağı pozitif)
ARRIVAL_TOLERANCE = 1.5             # metre (hedefe bu kadar yaklaşınca "vardım" de)

# Hedef koordinatları metreye önceden hesapla (BT içinde kullanmak için)
METERS_PER_DEGREE = 111320.0
TARGET_X = (HEDEF_LAT - BASLANGIC_LAT) * METERS_PER_DEGREE
TARGET_Y = (HEDEF_LON - BASLANGIC_LON) * METERS_PER_DEGREE * math.cos(math.radians(BASLANGIC_LAT))


def create_root() -> py_trees.behaviour.Behaviour:

    # ==========================================
    # 1. ANA YAPISI
    # ==========================================
    root = py_trees.composites.Parallel(
        name="DVL Transit Root",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    publishers_parallel = py_trees.composites.Parallel(
        name="Publishers",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    main_mission_sequence = py_trees.composites.Sequence("DVL Transit Mission", memory=True)

    one_shot_mission = py_trees.decorators.OneShot(
        name="Mission OneShot",
        child=main_mission_sequence,
        policy=py_trees.common.OneShotPolicy.ON_SUCCESSFUL_COMPLETION
    )

    # ==========================================
    # 2. PUBLISHERS (Blackboard'a Durum Yazanlar)
    # ==========================================
    mode2bb = common_behaviors.state.ToBlackboard(
        name="Mode2BB",
        topic_name="/vehicle/state",
        qos_profile=qos_profile_sensor_data
    )

    publishers_parallel.add_children([mode2bb])

    # ==========================================
    # 3. GUIDED MODA GEÇ
    # ==========================================
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

    # ==========================================
    # 4. ARM ET
    # ==========================================
    arm_request = SetBool.Request()
    arm_request.data = True
    arm_vehicle = py_trees_ros.service_clients.FromConstant(
        name="ArmVehicle",
        service_type=SetBool,
        service_name="/arm",
        service_request=arm_request
    )

    # ==========================================
    # 5. HEDEFE GİT (GTL Transporter Servisi)
    # ==========================================
    gps_request = GoToGpsTarget.Request()
    gps_request.baslangic_lat = BASLANGIC_LAT
    gps_request.baslangic_lon = BASLANGIC_LON
    gps_request.hedef_lat     = HEDEF_LAT
    gps_request.hedef_lon     = HEDEF_LON
    gps_request.target_depth  = HEDEF_DERINLIK

    go_to_target = py_trees_ros.service_clients.FromConstant(
        name="GoToGpsTarget",
        service_type=GoToGpsTarget,
        service_name="/compute_and_go_gps",
        service_request=gps_request
    )

    # ==========================================
    # 6. HEDEFE VARIŞ KONTROLÜ
    # ==========================================
    wait_arrival = behaviours.wait_for_arrival.WaitForArrival(
        name="WaitForArrival",
        target_x=TARGET_X,
        target_y=TARGET_Y,
        tolerance=ARRIVAL_TOLERANCE
    )

    # ==========================================
    # 6. AĞACI BİRLEŞTİR
    # ==========================================
    main_mission_sequence.add_children([
        retry_switch_guided,
        arm_vehicle,
        go_to_target,
        wait_arrival,
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

    print("DVL Transit Test BT Started... (Press CTRL+Z to stop)")
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
