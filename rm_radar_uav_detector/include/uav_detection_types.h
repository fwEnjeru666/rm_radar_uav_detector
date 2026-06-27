#pragma once

#include <string>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/TransformStamped.h>
#include <opencv2/core.hpp>
#include <rm_radar_msgs/DroneTrackData.h>
#include <rm_radar_msgs/UavDetection.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>

namespace rm_radarplugin
{

class Bbox2D
{
public:
  // Projected 2D bbox and optional ROI derived from the 3D lidar track.
  struct Bbox
  {
    bool is_valid{false};
    bool need_roi{false};
    cv::Point2f bbox_min_pt{-1.0f, -1.0f};
    cv::Point2f bbox_max_pt{-1.0f, -1.0f};
    cv::Point2f target_pt{-1.0f, -1.0f};
    cv::Rect bbox_area;
    cv::Rect roi_area;
    int roi_x{-1};
    int roi_y{-1};
    float expand_ratio{0.0f};
    ros::Time last_update_time;
  };

  Bbox bbox2d;

  void clear()
  {
    bbox2d = Bbox();
  }
};

struct ImageDetection
{
  // Latest camera-only detection cached by CameraDetector.
  std_msgs::Header header;
  geometry_msgs::Point pixel;
  geometry_msgs::Point ray;
  geometry_msgs::Point point3d;
  bool has_point3d{false};
  bool valid{false};
  double confidence{0.0};
  double error_x{0.0};
  double error_y{0.0};
};

struct UavDetectorContext
{
  // Per-image snapshot shared by camera-only and lidar-camera fusion detectors.
  std_msgs::Header image_header;
  cv::Mat intrinsics;
  cv::Mat dist_coeffs;
  int image_width{-1};
  int image_height{-1};

  bool debug_mode{false};
  bool need_roi{false};
  float expand_ratio{0.0f};
  double tf_timeout_s{0.02};
  double image_detection_timeout_s{0.5};
  double point_match_max_pixel_distance{8.0};
  double pixel_bias_u{0.0};
  double pixel_bias_v{0.0};
  std::string fixed_frame{"base_link"};
  tf2_ros::Buffer* tf_buffer{nullptr};
  sensor_msgs::PointCloud2ConstPtr track_cloud;

  bool camera_model_ready() const
  {
    return !intrinsics.empty() && image_width > 0 && image_height > 0;
  }

  geometry_msgs::TransformStamped getTransform(const std::string& source_frame_id,
                                               const ros::Time& source_stamp,
                                               const ros::Time& fallback_stamp = ros::Time(0)) const
  {
    // Use the advanced TF API only when both image and source timestamps are available.
    if (!image_header.stamp.isZero() && !source_stamp.isZero())
    {
      return tf_buffer->lookupTransform(image_header.frame_id,
                                        image_header.stamp,
                                        source_frame_id,
                                        source_stamp,
                                        fixed_frame,
                                        ros::Duration(tf_timeout_s));
    }

    // Fall back to the simple API for untimed data or callers that need Time(0).
    return tf_buffer->lookupTransform(image_header.frame_id,
                                      source_frame_id,
                                      fallback_stamp,
                                      ros::Duration(tf_timeout_s));
  }

  bool getTransform(const std::string& source_frame_id,
                    const ros::Time& source_stamp,
                    geometry_msgs::TransformStamped& transform,
                    const ros::Time& fallback_stamp = ros::Time(0)) const
  {
    // Non-throwing wrapper used by detection paths that should fail quietly.
    try
    {
      transform = getTransform(source_frame_id, source_stamp, fallback_stamp);
    }
    catch (const tf2::TransformException& ex)
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector] TF lookup failed: %s", ex.what());
      }
      return false;
    }

    return true;
  }

  ImageDetection image_detection;
  bool has_track{false};
  rm_radar_msgs::DroneTrackData track;
};

struct UavDetectorOutput
{
  rm_radar_msgs::UavDetection msg;
  bool has_bbox{false};
  Bbox2D::Bbox bbox;
  bool has_projected_image_point{false};
  cv::Point2f projected_image_point{-1.0f, -1.0f};
};

}  // namespace rm_radarplugin
