import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Point
from auv_interfaces.msg import DetectionArray, DetectedObject
from std_msgs.msg import Float32

TARGET_SIZE = 150

class StationKeepingTracker(Node):
    def __init__(self):
        super().__init__('station_keeping_tracker')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, "/camera/front", self.image_callback, 10)
        self.dist_sub = self.create_subscription(Float32, '/target_distance', self.distance_callback, 10) #########
        self.yolo_pub = self.create_publisher(DetectionArray, '/yolo_detections', 10)
        self.real_distance = None
        self.srv = self.create_service(SetBool, 'toggle_station_keeping', self.toggle_callback)
        self.is_active = False
        self.tracker = None

        self.get_logger().info('waitinggmmmq')

    def toggle_callback(self, request, response):
        self.is_active = request.data   #T or F
        if self.is_active:
            self.tracker = None 
            self.real_distance = None
            response.message = "active"
            self.get_logger().info(response.message)
        else:
            self.tracker = None
            response.message = "not active"
            self.get_logger().info(response.message)
        
        response.success = True
        return response

    def distance_callback(self, msg):  ######
        self.real_distance = msg.data

    def image_callback(self, msg):
        if not self.is_active:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"bridge error {e}")
            return
        height, width = cv_image.shape[:2]

        if self.tracker is None:  
            center_x, center_y = width // 2, height // 2

            x = center_x - (TARGET_SIZE // 2)
            y = center_y - (TARGET_SIZE // 2)
            
            bbox = (x, y, TARGET_SIZE, TARGET_SIZE)

            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(cv_image, bbox)
            self.get_logger().info("texture learned")
            return

        success, bbox = self.tracker.update(cv_image)

        if success:
            x, y, w, h = [int(v) for v in bbox]

            cx, cy = 320.0, 240.0
            
            f_len = 640.0 # Koordinatlari normalize etmek icin kameranin genisligi
            
            # Pikselleri oranlayarak (-0.5 ile 0.5 arasi) gonderiyoruz ki matris cildirmasin!
            p1 = Point(x=float((x - cx) / f_len),     y=float((y - cy) / f_len),     z=0.0)           
            p2 = Point(x=float((x + w - cx) / f_len), y=float((y - cy) / f_len),     z=0.0)       
            p3 = Point(x=float((x + w - cx) / f_len), y=float((y + h - cy) / f_len), z=0.0)   
            p4 = Point(x=float((x - cx) / f_len),     y=float((y + h - cy) / f_len), z=0.0)

            det = DetectedObject()
            det.class_name = "station_target" # Action Server bu hedefi arayacak
            
            if self.real_distance is not None:
                det.distance = float(self.real_distance)
            else:
                det.distance = 1.0
            
            det.yaw_angle = 0.0 # Yaw degismeyecek diye varsaydık
            det.keypoints = [p1, p2, p3, p4]

            msg_out = DetectionArray()
            msg_out.detections = [det]

            self.yolo_pub.publish(msg_out)
        else:
           self.get_logger().warn("target lost, stopping auv")
           empty_msg = DetectionArray()
           empty_msg.detections = [] # Ici bos array
           self.yolo_pub.publish(empty_msg)
           self.is_active = False

def main(args=None):
    rclpy.init(args=args)
    node = StationKeepingTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()