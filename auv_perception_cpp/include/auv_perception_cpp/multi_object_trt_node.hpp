#ifndef AUV_PERCEPTION_CPP__MULTI_OBJECT_TRT_NODE_HPP_
#define AUV_PERCEPTION_CPP__MULTI_OBJECT_TRT_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "auv_interfaces/msg/detection_array.hpp"
#include "auv_interfaces/msg/detected_object.hpp"
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

namespace auv_perception_cpp
{

struct ObjectProps {
  std::string name;
  std::vector<cv::Point3f> points_3d;
};

class MultiObjectTrtNode : public rclcpp::Node
{
public:
  explicit MultiObjectTrtNode(const rclcpp::NodeOptions & options);
  virtual ~MultiObjectTrtNode();

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  
  rclcpp::Publisher<auv_interfaces::msg::DetectionArray>::SharedPtr target_pub_;
  
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  bool has_camera_info_ = false;
  std::map<int, ObjectProps> object_library_;

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
};

}  

#endif 