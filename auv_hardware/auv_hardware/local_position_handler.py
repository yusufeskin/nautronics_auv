#!/usr/bin/env python3
from geometry_msgs.msg import Point


class LocalPositionHandler:

    def __init__(self, node, publisher):
        self.node = node
        self.publisher = publisher

    def handle_message(self, msg):
        point = Point()
        point.x = float(msg.x)
        point.y = float(msg.y)
        point.z = float(msg.z)
        self.publisher.publish(point)
