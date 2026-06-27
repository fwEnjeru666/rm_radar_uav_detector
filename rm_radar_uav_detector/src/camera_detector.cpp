#include "camera_detector.h"

#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace rm_radarplugin
{

void CameraDetector::updateDetection(const rm_radar_msgs::DroneDetection& msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!msg.ref_pt_valid || msg.header.frame_id.empty())
  {
    latest_detection_ = ImageDetection();
    return;
  }

  latest_detection_.header = msg.header;
  latest_detection_.pixel = msg.ref_pt;
  latest_detection_.ray = msg.ref_pt_camera;
  latest_detection_.point3d = msg.ref_pt_3d;
  latest_detection_.has_point3d = msg.ref_pt_3d_valid;
  latest_detection_.valid = true;
  latest_detection_.confidence = msg.confidence;
  latest_detection_.error_x = msg.error_x;
  latest_detection_.error_y = msg.error_y;
}

void CameraDetector::clear()
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_detection_ = ImageDetection();
}

ImageDetection CameraDetector::getLatestDetection() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return latest_detection_;
}

UavDetectorOutput CameraDetector::detect(const UavDetectorContext& context) const
{
  UavDetectorOutput output = projectDetectionToImage(context);
  output.msg = makeMessage(context.image_header, context.image_detection);
  return output;
}

UavDetectorOutput CameraDetector::projectDetectionToImage(const UavDetectorContext& context) const
{
  UavDetectorOutput output;
  const auto& detection = context.image_detection;
  if (!context.camera_model_ready())
  {
    return output;
  }

  if (!detection.valid || context.image_header.frame_id.empty() || detection.header.frame_id.empty() || !context.tf_buffer)
  {
    return output;
  }
  if (!detection.has_point3d)
  {
    if (context.debug_mode)
    {
      ROS_WARN_THROTTLE(1.0,
                        "[rm_radar_uav_detector][camera] detection has no 3D point; cannot reproject physical point");
    }
    return output;
  }

  if (!context.image_header.stamp.isZero() && !detection.header.stamp.isZero())
  {
    // Drop stale image detections before using their 3D point for reprojection.
    const double age_s = (context.image_header.stamp - detection.header.stamp).toSec();
    if (std::abs(age_s) > context.image_detection_timeout_s)
    {
      if (context.debug_mode)
      {
        ROS_WARN_THROTTLE(1.0,
                          "[rm_radar_uav_detector][camera] stale detection dt=%.3fs > %.3fs",
                          age_s,
                          context.image_detection_timeout_s);
      }
      return output;
    }
  }

  geometry_msgs::TransformStamped tf_msg;
  if (!context.getTransform(detection.header.frame_id,
                            detection.header.stamp,
                            tf_msg,
                            detection.header.stamp))
  {
    return output;
  }

  geometry_msgs::PointStamped ref_src;
  ref_src.header = detection.header;
  ref_src.point = detection.point3d;

  geometry_msgs::PointStamped ref_target;
  tf2::doTransform(ref_src, ref_target, tf_msg);
  if (ref_target.point.z <= 1e-4)
  {
    // Points behind the camera cannot be projected into the image plane.
    return output;
  }

  // Reproject the physical detection point into the current camera image.
  std::vector<cv::Point3f> object_points(1);
  object_points[0] = cv::Point3f(static_cast<float>(ref_target.point.x),
                                 static_cast<float>(ref_target.point.y),
                                 static_cast<float>(ref_target.point.z));
  std::vector<cv::Point2f> projected_points;
  cv::projectPoints(object_points,
                    cv::Vec3d::all(0.0),
                    cv::Vec3d::all(0.0),
                    context.intrinsics,
                    context.dist_coeffs,
                    projected_points);
  if (projected_points.size() != 1 || !std::isfinite(projected_points.front().x) ||
      !std::isfinite(projected_points.front().y))
  {
    return output;
  }

  const cv::Point2f image_point = projected_points.front();
  if (image_point.x < 0.0f || image_point.x >= static_cast<float>(context.image_width) || image_point.y < 0.0f ||
      image_point.y >= static_cast<float>(context.image_height))
  {
    if (context.debug_mode)
    {
      ROS_WARN_THROTTLE(1.0,
                        "[rm_radar_uav_detector][camera] projected detection out of image: u=%.1f v=%.1f",
                        image_point.x,
                        image_point.y);
    }
    return output;
  }

  output.has_projected_image_point = true;
  output.projected_image_point = image_point;
  return output;
}

rm_radar_msgs::UavDetection CameraDetector::makeMessage(const std_msgs::Header& header,
                                                        const ImageDetection& detection)
{
  rm_radar_msgs::UavDetection msg;
  msg.header = header;
  msg.valid = detection.valid;
  msg.confidence = detection.confidence;
  msg.has_vision = detection.valid;
  msg.error_x = detection.error_x;
  msg.error_y = detection.error_y;
  msg.has_2d = detection.valid;
  msg.pixel = detection.pixel;
  msg.has_3d = false;
  msg.tracking = false;
  return msg;
}

}  // namespace rm_radarplugin
