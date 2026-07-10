#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.executors import MultiThreadedExecutor

import py_trees
import py_trees_ros.trees
import py_trees.console as console

import subprocess

class RunLaunchFile(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, package: str, launch_file: str):
        super().__init__(name)
        self.package = package
        self.launch_file = launch_file
        self.process = None

    def update(self) -> py_trees.common.Status:
        if self.process is None:
            self.logger.debug(f"Starting launch file: {self.launch_file}")
            # Start the ROS 2 launch subprocess
            self.process = subprocess.Popen(
                ["ros2", "launch", self.package, self.launch_file]
            )
            return py_trees.common.Status.RUNNING

        # Check if process has finished
        retcode = self.process.poll()
        if retcode is None:
            return py_trees.common.Status.RUNNING
        
        # Process finished
        if retcode == 0:
            self.logger.debug(f"Launch file {self.launch_file} finished successfully")
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.debug(f"Launch file {self.launch_file} failed with code {retcode}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status: py_trees.common.Status):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

def create_root() -> py_trees.behaviour.Behaviour:
    # MAIN TREE STRUCTURE
    # We use a Sequence to run mission launch files one by one
    root = py_trees.composites.Sequence("Main Missions Sequence", memory=True)

    # -------------------------------------------------------------------------
    # ADD MISSIONS
    # -------------------------------------------------------------------------

    # --- 1. Gate Mission ---
    # This will run `ros2 launch auv_bringup gate_bt.launch.py`
    gate_mission = RunLaunchFile(
        name="Gate Mission Launch",
        package="auv_bringup",
        launch_file="gate_bt.launch.py"
    )
    root.add_child(gate_mission)

    # --- 2. Future Missions ---
    # Example for Torpedo:
    # torpedo_mission = RunLaunchFile(
    #     name="Torpedo Mission Launch",
    #     package="auv_bringup",
    #     launch_file="torpedo_bt.launch.py"
    # )
    # root.add_child(torpedo_mission)

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

    print("Starting Main Behavior Tree... (Press CTRL+Z to stop)")
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
