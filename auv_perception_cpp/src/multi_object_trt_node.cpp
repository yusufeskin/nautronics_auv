#include "auv_perception_cpp/multi_object_trt_node.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace auv_perception_cpp
{

MultiObjectTrtNode::MultiObjectTrtNode(const rclcpp::NodeOptions & options)
: Node("multi_object_trt_node", options)
{
  RCLCPP_INFO(this->get_logger(), "TensorRT Perception Node is starting...");

  std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("auv_perception_cpp");
  std::string config_path = pkg_share_dir + "/config/object_config.yaml";

  try {
    YAML::Node config = YAML::LoadFile(config_path);
    if (config["objects"]) {
      for (YAML::const_iterator it = config["objects"].begin(); it != config["objects"].end(); ++it) {
        int id = it->first.as<int>();
        std::string name = it->second["name"].as<std::string>();
        
        std::vector<cv::Point3f> pts;
        for (const auto& pt : it->second["points"]) {
          pts.push_back(cv::Point3f(pt[0].as<float>(), pt[1].as<float>(), pt[2].as<float>()));
        }
        
        object_library_[id] = {name, pts};
      }
      RCLCPP_INFO(this->get_logger(), "Object config başarıyla yüklendi! %zu hedef tanımlandı.", object_library_.size());
    }
  } catch (const YAML::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "YAML dosyası okunamadı: %s", e.what());
  }

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/image_raw", 
    10, 
    std::bind(&MultiObjectTrtNode::image_callback, this, std::placeholders::_1),
    sub_options
  );

  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera_info",
    10,
    std::bind(&MultiObjectTrtNode::camera_info_callback, this, std::placeholders::_1),
    sub_options
  );

  target_pub_ = this->create_publisher<auv_interfaces::msg::DetectionArray>("/yolo_detections", 10);
}

MultiObjectTrtNode::~MultiObjectTrtNode()
{
  RCLCPP_INFO(this->get_logger(), "TensorRT Node shutting down.");
}

void MultiObjectTrtNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  camera_matrix_ = (cv::Mat_<float>(3, 3) << 
    msg->k[0], msg->k[1], msg->k[2],
    msg->k[3], msg->k[4], msg->k[5],
    msg->k[6], msg->k[7], msg->k[8]);

  cv::Mat dist_double = cv::Mat(msg->d);
  dist_double.convertTo(dist_coeffs_, CV_32F);

  if (!has_camera_info_) {
    RCLCPP_INFO(this->get_logger(), "Kamera kalibrasyon parametreleri topic üzerinden başarıyla alındı!");
    has_camera_info_ = true;
  }
}

void MultiObjectTrtNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  if (!has_camera_info_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                         "Kamera bilgisi bekleniyor, PnP hesaplaması atlandı...");
    return;
  }

  try {
    cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
    
    std::vector<cv::Point2f> fake_2d_points = {
      cv::Point2f(100.0f, 100.0f),
      cv::Point2f(500.0f, 100.0f),
      cv::Point2f(500.0f, 300.0f),
      cv::Point2f(100.0f, 300.0f)
    };

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(
      object_library_[0].points_3d, 
      fake_2d_points,               
      camera_matrix_, 
      dist_coeffs_, 
      rvec, 
      tvec, 
      false, 
      cv::SOLVEPNP_ITERATIVE
    );

    if (success) {
      double distance = tvec.at<double>(2); 
      RCLCPP_INFO(this->get_logger(), "Dinamik PnP Başarılı! Hedef: %s | Mesafe: %.2f metre", 
                  object_library_[0].name.c_str(), distance);

      auv_interfaces::msg::DetectionArray det_array;
      det_array.header = msg->header; 
      
      auv_interfaces::msg::DetectedObject obj_msg;
      obj_msg.class_id = 0; 
      obj_msg.class_name = object_library_[0].name;
      obj_msg.distance = distance;
      
      cv::Mat rmat;
      cv::Rodrigues(rvec, rmat);
      double yaw = atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
      obj_msg.yaw_angle = yaw;

      det_array.detections.push_back(obj_msg);
      
      target_pub_->publish(det_array);

    } else {
      RCLCPP_WARN(this->get_logger(), "Dinamik PnP Çözülemedi!");
    }

  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

}  

RCLCPP_COMPONENTS_REGISTER_NODE(auv_perception_cpp::MultiObjectTrtNode)