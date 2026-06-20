#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3
import py_trees
import py_trees_ros.trees
import py_trees.console as console

import behaviours.set_attitude_action
import behaviours.attitude


def create_yaw_step(increment: float, step_name: str) -> py_trees.composites.Parallel:

    parallel = py_trees.composites.Parallel(
        name=step_name,
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )

    set_att = behaviours.set_attitude_action.SetAttitudeAction(
        name=f"Send +{increment}°",
        topic="/target_attitude",
        yaw_increment=increment,
        target_roll=0.0,
        target_pitch=0.0
    )

    checker = behaviours.attitude.AttitudeCheckerCondition(
        name=f"Check +{increment}°",
        topic="/current_attitude",
        tolerance=2.0
    )

    parallel.add_children([set_att, checker])
    return parallel


def create_root() -> py_trees.behaviour.Behaviour:

    attitude2bb = py_trees_ros.subscribers.ToBlackboard(
        name="Attitude2BB",
        topic_name="/current_attitude",
        topic_type=Vector3,
        blackboard_variables={'current_yaw': 'z'},
        qos_profile=qos_profile_sensor_data
    )

    mission = py_trees.composites.Sequence(
        name="Yaw Mission Sequence",
        memory=True
    )

    mission.add_child(create_yaw_step(90.0, "Step 1: +90°"))
    mission.add_child(create_yaw_step(90.0, "Step 2: +90°"))
    mission.add_child(create_yaw_step(90.0, "Step 3: +90°"))
    mission.add_child(create_yaw_step(90.0, "Step 4: +90°"))

    root = py_trees.composites.Parallel(
        name="Root",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()  # mission bitince root da biter
    )
    root.add_children([attitude2bb, mission])

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
        console.logerror("Setup failed: {}".format(str(e)))
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)
    except KeyboardInterrupt:
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)

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