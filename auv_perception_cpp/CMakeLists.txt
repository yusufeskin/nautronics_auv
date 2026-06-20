cmake_minimum_required(VERSION 3.8)
project(auv_perception_cpp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(auv_interfaces REQUIRED)

add_library(multi_object_trt_component SHARED
  src/multi_object_trt_node.cpp
)

target_include_directories(multi_object_trt_component PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(multi_object_trt_component
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  yaml-cpp
  ament_index_cpp
  auv_interfaces
)

target_link_libraries(multi_object_trt_component
  ${OpenCV_LIBS}
  yaml-cpp
)

rclcpp_components_register_node(multi_object_trt_component
  PLUGIN "auv_perception_cpp::MultiObjectTrtNode"
  EXECUTABLE multi_object_trt_node_exec
)

install(TARGETS
  multi_object_trt_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
# Config klasörünü de sisteme kopyalamasını söylüyoruz
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()