import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from my_custom_interface.msg import GateDetectionArray, GateDetection, Keypoint


class Detector(Node):
    def _init_(self):
        super()._init_('detector')
        self.declare_parameter('model_path', '') #Default olarak boş bırakılmış 
        self.model = YOLO(self.get_parameter('model_path').value)
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/front', self.on_image,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(GateDetectionArray, '/gate/detections', 10)

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        r = self.model(frame, verbose=False)[0]
        out = GateDetectionArray(header=msg.header)   # görüntünün stamp'i
        kps = r.keypoints

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            g = GateDetection()
            g.class_name = r.names[int(box.cls)]
            g.score = float(box.conf)
            g.cx, g.cy = (x1 + x2) / 2, (y1 + y2) / 2
            g.width, g.height = x2 - x1, y2 - y1

            if kps is not None:
                xy = kps.xy[i].tolist()
                cf = kps.conf[i].tolist() if kps.conf is not None else [1.0] * len(xy)
                g.keypoints = [Keypoint(x=x, y=y, conf=c) for (x, y), c in zip(xy, cf)]

            out.detections.append(g)

        self.pub.publish(out)   # tespit yoksa da boş mesaj gider


def main():
    rclpy.init()
    rclpy.spin(Detector())