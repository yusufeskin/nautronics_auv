#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
from behaviours.set_attitude_action import SetAttitudeAction
from behaviours.attitude import AttitudeCheckerCondition

def create_yaw_step(increment: float, step_name: str) -> py_trees.composites.Parallel:
    """
    Belirtilen açı kadar dönüş yapılması için hedef açıyı gönderen ve 
    ulaşılıp ulaşılmadığını kontrol eden paralel bir behaviour tree adımı oluşturur.
    """
    parallel = py_trees.composites.Parallel(
        name=step_name,
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )

    set_att = SetAttitudeAction(
        name=f"Send +{increment}°",
        topic="/target_attitude",
        yaw_increment=increment,
        target_roll=0.0,
        target_pitch=0.0
    )

    checker = AttitudeCheckerCondition(
        name=f"Check +{increment}°",
        tolerance=2.0
    )

    parallel.add_children([set_att, checker])
    return parallel
