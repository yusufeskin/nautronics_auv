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
    "/image_raw", 10, std::bind(&MultiObjectTrtNode::image_callback, this, std::placeholders::_1), sub_options);

  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera_info", 10, std::bind(&MultiObjectTrtNode::camera_info_callback, this, std::placeholders::_1), sub_options);

  target_pub_ = this->create_publisher<auv_interfaces::msg::DetectionArray>("/yolo_detections", 10);
}

MultiObjectTrtNode::~MultiObjectTrtNode()
{
  RCLCPP_INFO(this->get_logger(), "TensorRT Node shutting down.");
}

void MultiObjectTrtNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  camera_matrix_ = (cv::Mat_<float>(3, 3) << 
    msg->k[0], msg->k[1], msg->k[2], msg->k[3], msg->k[4], msg->k[5], msg->k[6], msg->k[7], msg->k[8]);
  cv::Mat dist_double = cv::Mat(msg->d);
  dist_double.convertTo(dist_coeffs_, CV_32F);

  if (!has_camera_info_) {
    RCLCPP_INFO(this->get_logger(), "Kamera kalibrasyon parametreleri topic üzerinden başarıyla alındı!");
    has_camera_info_ = true;
  }
}

// ==============================================================================
// GERÇEK TENSORRT KODLARI BURAYA GELECEK
// Şu an boş döndürüyor ki derlemede hata vermesin. C++ TensorRT API'sini 
// buraya bağlayıp InferenceResult listesi döndüreceksin.
// ==============================================================================
std::vector<InferenceResult> MultiObjectTrtNode::run_tensorrt_inference(const cv::Mat& frame)
{
  std::vector<InferenceResult> detections;
  return detections;
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

    std::vector<InferenceResult> ai_results = run_tensorrt_inference(frame);

    auv_interfaces::msg::DetectionArray det_array;
    det_array.header = msg->header;

    for (const auto& det : ai_results) {
      
      if (object_library_.find(det.class_id) == object_library_.end() || det.keypoints.size() < 4) {
        continue;
      }

      auv_interfaces::msg::DetectedObject obj_msg;
      obj_msg.class_id = det.class_id;
      obj_msg.class_name = object_library_[det.class_id].name;
      obj_msg.confidence = det.confidence;

      for (size_t i = 0; i < 4; ++i) {
        obj_msg.keypoints[i].x = det.keypoints[i].x;
        obj_msg.keypoints[i].y = det.keypoints[i].y;
        obj_msg.keypoints[i].z = 0.0;
      }

      cv::Mat rvec, tvec;
      bool success = cv::solvePnP(
        object_library_[det.class_id].points_3d, 
        det.keypoints,               
        camera_matrix_, 
        dist_coeffs_, 
        rvec, 
        tvec, 
        false, 
        cv::SOLVEPNP_ITERATIVE
      );

      if (success) {
        obj_msg.distance = tvec.at<double>(2); 
        
        cv::Mat rmat;
        cv::Rodrigues(rvec, rmat);
        
        double yaw = atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
        obj_msg.yaw_angle = yaw;

      } else {
        obj_msg.distance = -1.0;
        obj_msg.yaw_angle = 0.0;
      }

      det_array.detections.push_back(obj_msg);
    }

    target_pub_->publish(det_array);

  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

}  

RCLCPP_COMPONENTS_REGISTER_NODE(auv_perception_cpp::MultiObjectTrtNode)