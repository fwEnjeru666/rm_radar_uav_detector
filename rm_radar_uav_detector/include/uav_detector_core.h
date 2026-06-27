#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <opencv2/core.hpp>
#include <rm_radar_msgs/DroneDetection.h>
#include <rm_radar_msgs/DroneTrackData.h>
#include <rm_radar_msgs/UavDetection.h>
#include <rm_radar_uav_detector/UavDetectorConfig.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <tf2_ros/transform_listener.h>

#include "camera_detector.h"
#include "lidar_camera_fusion_detector.h"
#include "uav_detection_types.h"

namespace rm_radarplugin
{

class UavDetectorCore : public nodelet::Nodelet
{
public:
  UavDetectorCore() = default;
  ~UavDetectorCore() override = default;

  void onInit() override;

private:
  // Runtime parameters are copied under lock once per image callback.
  struct RuntimeParams
  {
    double timeout_s{0.2};
    double tf_timeout_s{0.02};
    double image_detection_timeout_s{0.5};
    double point_match_max_pixel_distance{8.0};
    bool debug_mode{false};
    bool need_roi{false};
    float expand_ratio{0.0f};
    double pixel_bias_u{0.0};
    double pixel_bias_v{0.0};
    std::string fixed_frame{"base_link"};
  };

  static constexpr std::size_t kTrackBufferMaxSize = 200;

  // ROS callbacks update independent caches; imageCallback performs the fusion decision.
  void lidarDetectionCallback(const rm_radar_msgs::DroneTrackData::ConstPtr& msg);
  void imageDetectionCallback(const rm_radar_msgs::DroneDetection::ConstPtr& msg);
  void imageCallback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info);
  void reconfigCB(rm_radar_uav_detector::UavDetectorConfig& config, uint32_t level);

  void updateCameraModel(const sensor_msgs::CameraInfoConstPtr& info);
  bool getNearestTrack(const ros::Time& image_stamp, rm_radar_msgs::DroneTrackData& track) const;
  RuntimeParams getRuntimeParams() const
  {
    RuntimeParams params;
    std::lock_guard<std::mutex> lock(param_mutex_);
    params.timeout_s = timeout_s_;
    params.tf_timeout_s = tf_timeout_s_;
    params.image_detection_timeout_s = image_detection_timeout_s_;
    params.point_match_max_pixel_distance = point_match_max_pixel_distance_;
    params.debug_mode = debug_mode_;
    params.need_roi = need_roi_;
    params.expand_ratio = expand_ratio_;
    params.pixel_bias_u = pixel_bias_u_;
    params.pixel_bias_v = pixel_bias_v_;
    params.fixed_frame = fixed_frame_;
    return params;
  }

  UavDetectorContext buildContext(const std_msgs::Header& image_header,
                                  const RuntimeParams& params,
                                  bool has_track,
                                  const rm_radar_msgs::DroneTrackData& track) const;
  // Accept untimed detections, otherwise enforce camera-image freshness.
  static bool hasValidCameraDetection(const UavDetectorContext& context);
  void updateProjectedImagePoint(const UavDetectorOutput& output)
  {
    std::lock_guard<std::mutex> lock(projected_image_point_mutex_);
    projected_image_point_ = output.projected_image_point;
    projected_image_point_valid_ = output.has_projected_image_point;
  }

  void publishOutput(const UavDetectorOutput& output)
  {
    if (output.msg.valid)
    {
      uav_detection_pub_.publish(output.msg);
    }
  }

  void clearDroneBbox()
  {
    std::lock_guard<std::mutex> lock(bbox_mutex_);
    drone_bbox_.clear();
  }

  void publishRoiImage(const cv_bridge::CvImageConstPtr& image, const Bbox2D::Bbox& bbox, bool debug_mode);
  void drawBbox(const cv_bridge::CvImageConstPtr& temp, const Bbox2D::Bbox* bbox, const std::string& status_text);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  std::shared_ptr<image_transport::ImageTransport> it_;
  image_transport::CameraSubscriber image_sub_;
  image_transport::Publisher debug_image_pub_;
  image_transport::Publisher roi_image_pub_;
  ros::Subscriber track_sub_;
  ros::Subscriber image_detection_sub_;
  ros::Publisher uav_detection_pub_;

  mutable tf2_ros::Buffer tf_buffer_{ros::Duration(10.0)};
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<dynamic_reconfigure::Server<rm_radar_uav_detector::UavDetectorConfig>> cfg_server_;
  dynamic_reconfigure::Server<rm_radar_uav_detector::UavDetectorConfig>::CallbackType cfg_cb_;

  mutable std::mutex param_mutex_;
  mutable std::mutex track_mutex_;
  mutable std::mutex camera_model_mutex_;
  mutable std::mutex bbox_mutex_;
  mutable std::mutex projected_image_point_mutex_;

  std::deque<rm_radar_msgs::DroneTrackData> track_buffer_;
  cv::Point2f projected_image_point_{-1.0f, -1.0f};
  bool projected_image_point_valid_{false};

  cv::Mat intrinsics_;
  cv::Mat dist_coeffs_;
  sensor_msgs::CameraInfoConstPtr camera_info_;
  bool camera_model_initialized_{false};
  int image_width_{-1};
  int image_height_{-1};

  std::string image_topic_;
  std::string track_topic_;
  std::string image_detection_topic_;
  std::string roi_topic_;
  std::string debug_topic_{"debugimage"};
  std::string uav_detection_topic_{"/rm_radar_uav_detector/uav_detection"};
  std::string fixed_frame_{"base_link"};

  double timeout_s_{0.2};
  double tf_timeout_s_{0.02};
  double image_detection_timeout_s_{0.5};
  double point_match_max_pixel_distance_{8.0};
  int line_thickness_{2};
  bool debug_mode_{false};
  bool disable_{false};
  bool draw_bbox_{true};
  bool draw_roi_{true};
  double pixel_bias_u_{0.0};
  double pixel_bias_v_{0.0};
  bool need_roi_{false};
  float expand_ratio_{0.0f};

  Bbox2D drone_bbox_;
  CameraDetector camera_detector_;
  LidarCameraFusionDetector lidar_camera_fusion_detector_;
};

}  // namespace rm_radarplugin
