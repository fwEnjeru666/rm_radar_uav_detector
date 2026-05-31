cmake_minimum_required(VERSION 3.0.2)
project(rm_radar_cam_detector)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  image_transport
  nodelet
  rm_radar_msgs
  roscpp
  sensor_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)

generate_dynamic_reconfigure_options(
  cfg/CamDetector.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    cv_bridge
    dynamic_reconfigure
    image_transport
    nodelet
    rm_radar_msgs
    roscpp
    sensor_msgs
    tf2
    tf2_geometry_msgs
    tf2_ros
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/rm_radar_cam_detector.cpp
)

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
