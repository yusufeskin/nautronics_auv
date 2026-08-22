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
import behaviours.wait_for_depth

from auv_interfaces.srv import SetVehicleMode, GoToGpsTarget
from std_srvs.srv import SetBool

# ==========================================
# KOORDİNATLAR
# ==========================================
# Baslangic: pixhawk_bridge2.py'deki POOL_ORIGIN_LAT/LON ile ayni olmali.
BASLANGIC_LAT = 40.7580166667
BASLANGIC_LON = 29.9013972222

MISSION_DEPTH = 1.0                 # metre, gorev boyunca sabit
ARRIVAL_TOLERANCE = 0.4             # metre
DEPTH_ARRIVAL_TOLERANCE = 0.2       # metre

METERS_PER_DEGREE = 111320.0

# Yarisma gunu Google Earth'ten okunan noktalari sirayla buraya ekleyin.
# Baslangic ayrica WAYPOINTS'e eklenmez, arac zaten oradan basliyor.
WAYPOINTS = [
    {"lat": 40.7579750000, "lon": 29.9012694444, "depth": MISSION_DEPTH},
    {"lat": 40.7580416667, "lon": 29.9013500000, "depth": MISSION_DEPTH},
]


def latlon_to_local(lat, lon):
    x = (lat - BASLANGIC_LAT) * METERS_PER_DEGREE
    y = (lon - BASLANGIC_LON) * METERS_PER_DEGREE * math.cos(math.radians(BASLANGIC_LAT))
    return x, y


def create_leg(index, waypoint, tolerance):
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
        tolerance=tolerance
    )

    return py_trees.composites.Sequence(
        name=f"Leg{index}",
        memory=True,
        children=[go_to_target, wait_arrival]
    )


def create_dive_step(depth):
    dive_request = GoToGpsTarget.Request()
    dive_request.baslangic_lat = BASLANGIC_LAT
    dive_request.baslangic_lon = BASLANGIC_LON
    dive_request.hedef_lat = BASLANGIC_LAT   # yatayda yerinde kal (local x,y = 0,0)
    dive_request.hedef_lon = BASLANGIC_LON
    dive_request.target_depth = depth

    dive_to_depth = py_trees_ros.service_clients.FromConstant(
        name="DiveToDepth",
        service_type=GoToGpsTarget,
        service_name="/compute_and_go_gps",
        service_request=dive_request
    )

    wait_depth = behaviours.wait_for_depth.WaitForDepth(
        name="WaitForDiveDepth",
        target_depth=depth,
        tolerance=DEPTH_ARRIVAL_TOLERANCE
    )

    return py_trees.composites.Sequence(
        name="DiveInPlace",
        memory=True,
        children=[dive_to_depth, wait_depth]
    )


def create_root() -> py_trees.behaviour.Behaviour:

    root = py_trees.composites.Parallel(
        name="Teknofest DVL Mission Root",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    publishers_parallel = py_trees.composites.Parallel(
        name="Publishers",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    main_mission_sequence = py_trees.composites.Sequence("Teknofest DVL Mission", memory=True)

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

    disarm_request = SetBool.Request()
    disarm_request.data = False
    disarm_vehicle = py_trees_ros.service_clients.FromConstant(
        name="DisarmVehicle",
        service_type=SetBool,
        service_name="/arm",
        service_request=disarm_request
    )

    legs = [create_leg(i, wp, ARRIVAL_TOLERANCE) for i, wp in enumerate(WAYPOINTS, start=1)]
    dive_step = create_dive_step(MISSION_DEPTH)

    main_mission_sequence.add_children([
        retry_switch_guided,
        arm_vehicle,
        dive_step,
        *legs,
        disarm_vehicle,
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

    print("Teknofest DVL Mission BT Started... (Press CTRL+Z to stop)")
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
