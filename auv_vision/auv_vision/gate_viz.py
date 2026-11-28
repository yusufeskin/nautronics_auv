import rclpy
from rclpy.node import Node
from my_custom_interface.msg import GateDetectionArray
from foxglove_msgs.msg import (ImageAnnotations, PointsAnnotation,
                               TextAnnotation, Point2, Color)


def rgba(r, g, b, a=1.0):
    return Color(r=r, g=g, b=b, a=a)


class GateViz(Node):
    def _init_(self):
        super()._init_('gate_viz')
        self.create_subscription(GateDetectionArray, '/gate/detections', self.cb, 10)
        self.pub = self.create_publisher(ImageAnnotations, '/gate/annotations', 10)

    def cb(self, msg):
        if self.pub.get_subscription_count() == 0:
            return
        t = msg.header.stamp
        ann = ImageAnnotations()

        for g in msg.detections:
            x1, y1 = g.cx - g.width / 2, g.cy - g.height / 2
            x2, y2 = g.cx + g.width / 2, g.cy + g.height / 2
            ann.points.append(PointsAnnotation(
                timestamp=t, type=PointsAnnotation.LINE_LOOP,
                points=[Point2(x=x1, y=y1), Point2(x=x2, y=y1),
                        Point2(x=x2, y=y2), Point2(x=x1, y=y2)],
                outline_color=rgba(0.0, 1.0, 0.3), thickness=2.0))

            visible = [k for k in g.keypoints if k.conf > 0.5]
            if visible:
                ann.points.append(PointsAnnotation(
                    timestamp=t, type=PointsAnnotation.POINTS,
                    points=[Point2(x=k.x, y=k.y) for k in visible],
                    outline_color=rgba(1.0, 0.2, 0.2), thickness=6.0))  # nokta çapı

            ann.texts.append(TextAnnotation(
                timestamp=t, position=Point2(x=x1, y=max(y1 - 4.0, 0.0)),
                text=f'{g.class_name} {g.score:.2f}', font_size=14.0,
                text_color=rgba(1.0, 1.0, 1.0),
                background_color=rgba(0.0, 0.0, 0.0, 0.6)))

        self.pub.publish(ann)


def main():
    rclpy.init()
    rclpy.spin(GateViz())


