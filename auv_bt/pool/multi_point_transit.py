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
import common_behaviors.state
import behaviours.wait_for_arrival

from auv_interfaces.srv import SetVehicleMode, GoToGpsTarget
from std_srvs.srv import SetBool

# ==========================================
# KOORDİNATLAR
# ==========================================
BASLANGIC_LAT = 39.856955555555555   # Havuz orijini (SET_GPS_GLOBAL_ORIGIN ile ayni nokta)
BASLANGIC_LON = 32.691258333333334

ARRIVAL_TOLERANCE = 0.4            # metre

# Sirayla gidilecek hedefler. Her biri BASLANGIC_LAT/LON referansindan hesaplanir.
WAYPOINTS = [
    {"lat": 39.85703611111111, "lon": 32.69118055555556, "depth": 0.5},
    # {"lat": ..., "lon": ..., "depth": ...},
]

METERS_PER_DEGREE = 111320.0


def latlon_to_local(lat, lon):
    x = (lat - BASLANGIC_LAT) * METERS_PER_DEGREE
    y = (lon - BASLANGIC_LON) * METERS_PER_DEGREE * math.cos(math.radians(BASLANGIC_LAT))
    return x, y


def create_leg(index, waypoint):
    target_x, target_y = latlon_to_local(waypoint["lat"], waypoint["lon"])

    gps_request = GoToGpsTarget.Request()
    gps_request.baslangic_lat = BASLANGIC_LAT
    gps_request.baslangic_lon = BASLANGIC_LON
    gps_request.hedef_lat = waypoint["lat"]
    gps_request.hedef_lon = waypoint["lon"]
    gps_request.target_depth = waypoint["depth"]

    go_to_target = py_trees_ros.service_clients.FromConstant(
        name=f"GoToWaypoint{index}",
        service_type=GoToGpsTarget,
        service_name="/compute_and_go_gps",
        service_request=gps_request
    )

    wait_arrival = behaviours.wait_for_arrival.WaitForArrival(
        name=f"WaitForArrival{index}",
        target_x=target_x,
        target_y=target_y,
        tolerance=ARRIVAL_TOLERANCE
    )

    return py_trees.composites.Sequence(
        name=f"Leg{index}",
        memory=True,
        children=[go_to_target, wait_arrival]
    )


def create_root() -> py_trees.behaviour.Behaviour:

    root = py_trees.composites.Parallel(
        name="Multi Point Transit Root",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    publishers_parallel = py_trees.composites.Parallel(
        name="Publishers",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    main_mission_sequence = py_trees.composites.Sequence("Multi Point Mission", memory=True)

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
    publishers_parallel.add_children([mode2bb])

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

    legs = [create_leg(i, waypoint) for i, waypoint in enumerate(WAYPOINTS, start=1)]

    main_mission_sequence.add_children([
        retry_switch_guided,
        arm_vehicle,
        *legs,
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

    print("Multi Point Transit BT Started... (Press CTRL+Z to stop)")
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
