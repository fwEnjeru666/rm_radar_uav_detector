cmake_minimum_required(VERSION 3.0.2)
project(rm_radar_img_proc)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
add_compile_options(-O3 -march=native -pipe)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        roslib
        dynamic_reconfigure
        std_msgs
        sensor_msgs
        image_transport
        cv_bridge
        nodelet
        rm_radar_msgs
        rm_msgs
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui)
find_package(OpenVINO REQUIRED COMPONENTS runtime)

generate_dynamic_reconfigure_options(
  cfg/Armor.cfg
  cfg/Draw.cfg
  cfg/Preprocess.cfg
)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
        CATKIN_DEPENDS
        roscpp
        roslib
        dynamic_reconfigure
        std_msgs
        sensor_msgs
        image_transport
        cv_bridge
        nodelet
        rm_radar_msgs
        rm_msgs
        DEPENDS OpenCV OpenVINO
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} 
        src/rm_radar_imgProc.cpp
        src/tools.cpp
        src/preprocess.cpp
        src/detect.cpp
        src/yolo_detector/yolo.cpp
        src/yolo_detector/models/yolo26.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE RM_RADAR_IMG_PROC_USE_OPENVINO=1)

target_link_libraries(
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        openvino::runtime
)

add_executable(${PROJECT_NAME}_find_ref_pt
        src/find_ref_pt.cpp
)
set_target_properties(${PROJECT_NAME}_find_ref_pt PROPERTIES OUTPUT_NAME find_ref_pt)

target_link_libraries(
        ${PROJECT_NAME}_find_ref_pt
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
)