#include "lidar_camera_fusion_detector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace rm_radarplugin
{

bool LidarCameraFusionDetector::buildBbox(const std::vector<cv::Point2f>& projected_points,
                                          const cv::Point2f& projected_target_point,
                                          bool debug_mode,
                                          bool need_roi,
                                          float expand_ratio,
                                          int image_width,
                                          int image_height,
                                          Bbox2D::Bbox& bbox)
{
  bbox = Bbox2D::Bbox();
  if (projected_points.empty())
  {
    if (debug_mode)
    {
      ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] projected points empty.");
    }
    return false;
  }

  float bbox_min_x = std::numeric_limits<float>::max();
  float bbox_min_y = std::numeric_limits<float>::max();
  float bbox_max_x = std::numeric_limits<float>::lowest();
  float bbox_max_y = std::numeric_limits<float>::lowest();

  for (const auto& point : projected_points)
  {
    if (!std::isfinite(point.x) || !std::isfinite(point.y))
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] projected point is NaN/Inf.");
      }
      return false;
    }
    bbox_min_x = std::min(bbox_min_x, point.x);
    bbox_min_y = std::min(bbox_min_y, point.y);
    bbox_max_x = std::max(bbox_max_x, point.x);
    bbox_max_y = std::max(bbox_max_y, point.y);
  }

  if (bbox_max_x <= bbox_min_x || bbox_max_y <= bbox_min_y)
  {
    if (debug_mode)
    {
      ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] invalid bbox min/max after projection.");
    }
    return false;
  }

  const cv::Rect2f raw_rect(cv::Point2f(bbox_min_x, bbox_min_y), cv::Point2f(bbox_max_x, bbox_max_y));
  const cv::Rect2f image_rect(0.0f, 0.0f, static_cast<float>(image_width), static_cast<float>(image_height));
  // Clip the projected 3D box to the image before publishing bbox/ROI.
  const cv::Rect2f clipped_rect = raw_rect & image_rect;
  if (clipped_rect.width <= 0.0f || clipped_rect.height <= 0.0f)
  {
    if (debug_mode)
    {
      ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] bbox has no overlap with image.");
    }
    return false;
  }

  bbox.is_valid = true;
  bbox.need_roi = need_roi;
  bbox.bbox_min_pt = cv::Point2f(clipped_rect.x, clipped_rect.y);
  bbox.bbox_max_pt = cv::Point2f(clipped_rect.x + clipped_rect.width, clipped_rect.y + clipped_rect.height);
  bbox.target_pt = projected_target_point;
  bbox.bbox_area = cv::Rect(static_cast<int>(std::floor(clipped_rect.x)),
                            static_cast<int>(std::floor(clipped_rect.y)),
                            static_cast<int>(std::ceil(clipped_rect.width)),
                            static_cast<int>(std::ceil(clipped_rect.height)));
  bbox.bbox_area &= cv::Rect(0, 0, image_width, image_height);
  bbox.expand_ratio = expand_ratio;
  bbox.last_update_time = ros::Time::now();

  if (need_roi && bbox.bbox_area.area() > 0)
  {
    // ROI is an expanded bbox used by downstream image processing.
    const float expand_w = bbox.bbox_area.width * expand_ratio;
    const float expand_h = bbox.bbox_area.height * expand_ratio;
    const int roi_x = static_cast<int>(std::floor(bbox.bbox_area.x - expand_w * 0.5f));
    const int roi_y = static_cast<int>(std::floor(bbox.bbox_area.y - expand_h * 0.5f));
    const int roi_w = static_cast<int>(std::ceil(bbox.bbox_area.width + expand_w));
    const int roi_h = static_cast<int>(std::ceil(bbox.bbox_area.height + expand_h));
    bbox.roi_area = cv::Rect(roi_x, roi_y, roi_w, roi_h) & cv::Rect(0, 0, image_width, image_height);
    bbox.roi_x = bbox.roi_area.x;
    bbox.roi_y = bbox.roi_area.y;
  }

  return true;
}

UavDetectorOutput LidarCameraFusionDetector::detect(const UavDetectorContext& context) const
{
  UavDetectorOutput output;
  const auto& track = context.track;
  const auto& image_detection = context.image_detection;

  output.msg.header = context.image_header;
  output.msg.header.frame_id = track.header.frame_id;
  output.msg.valid = context.has_track && track.tracking;
  output.msg.confidence = image_detection.valid ? image_detection.confidence : 1.0;
  output.msg.has_vision = image_detection.valid;
  output.msg.error_x = image_detection.valid ? image_detection.error_x : 0.0;
  output.msg.error_y = image_detection.valid ? image_detection.error_y : 0.0;
  output.msg.has_3d = context.has_track && track.tracking;
  output.msg.tracking = context.has_track && track.tracking;
  output.msg.position = track.position;

  if (!context.has_track || !track.tracking)
  {
    return output;
  }
  if (!context.camera_model_ready())
  {
    return output;
  }

  if (track.header.frame_id.empty() || context.image_header.frame_id.empty() || !context.tf_buffer)
  {
    if (context.debug_mode)
    {
      ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] empty frame id or tf buffer.");
    }
    return output;
  }

  geometry_msgs::TransformStamped tf_msg;
  if (!context.getTransform(track.header.frame_id,
                            track.header.stamp,
                            tf_msg))
  {
    return output;
  }

  std::vector<cv::Point3f> object_points;
  object_points.reserve(track.aabb_points.size());
  // Transform every AABB corner into the camera frame before projection.
  for (const auto& point : track.aabb_points)
  {
    geometry_msgs::PointStamped src;
    src.header = track.header;
    src.point = point;
    geometry_msgs::PointStamped dst;
    tf2::doTransform(src, dst, tf_msg);
    if (dst.point.z <= 1e-4)
    {
      if (context.debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] AABB point behind camera.");
      }
      return output;
    }
    object_points.emplace_back(static_cast<float>(dst.point.x),
                               static_cast<float>(dst.point.y),
                               static_cast<float>(dst.point.z));
  }

  if (object_points.empty())
  {
    if (context.debug_mode)
    {
      ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] track AABB points empty.");
    }
    return output;
  }

  std::vector<cv::Point2f> projected_points;
  // With points already in camera frame, projectPoints uses zero rvec/tvec.
  cv::projectPoints(object_points,
                    cv::Vec3d::all(0.0),
                    cv::Vec3d::all(0.0),
                    context.intrinsics,
                    context.dist_coeffs,
                    projected_points);
  if (projected_points.size() != object_points.size())
  {
    if (context.debug_mode)
    {
      ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] projected point count mismatch.");
    }
    return output;
  }

  geometry_msgs::PointStamped center_src;
  center_src.header = track.header;
  center_src.point = track.position;
  geometry_msgs::PointStamped center_dst;
  tf2::doTransform(center_src, center_dst, tf_msg);
  if (center_dst.point.z <= 1e-4)
  {
    if (context.debug_mode)
    {
      ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] track center behind camera.");
    }
    return output;
  }

  std::vector<cv::Point3f> center_points(1);
  center_points[0] = cv::Point3f(static_cast<float>(center_dst.point.x),
                                 static_cast<float>(center_dst.point.y),
                                 static_cast<float>(center_dst.point.z));
  std::vector<cv::Point2f> projected_center;
  cv::projectPoints(center_points,
                    cv::Vec3d::all(0.0),
                    cv::Vec3d::all(0.0),
                    context.intrinsics,
                    context.dist_coeffs,
                    projected_center);
  if (projected_center.size() != 1 || !std::isfinite(projected_center.front().x) ||
      !std::isfinite(projected_center.front().y))
  {
    if (context.debug_mode)
    {
      ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] failed to project track center.");
    }
    return output;
  }

  for (auto& point : projected_points)
  {
    point.x += static_cast<float>(context.pixel_bias_u);
    point.y += static_cast<float>(context.pixel_bias_v);
  }
  cv::Point2f projected_target_point = projected_center.front();
  projected_target_point.x += static_cast<float>(context.pixel_bias_u);
  projected_target_point.y += static_cast<float>(context.pixel_bias_v);

  Bbox2D::Bbox bbox;
  if (!buildBbox(projected_points,
                 projected_target_point,
                 context.debug_mode,
                 context.need_roi,
                 context.expand_ratio,
                 context.image_width,
                 context.image_height,
                 bbox))
  {
    return output;
  }

  output.has_bbox = true;
  output.bbox = bbox;
  output.msg.valid = true;
  output.msg.has_3d = true;
  output.msg.tracking = true;
  output.msg.header.frame_id = track.header.frame_id;
  return output;
}

bool LidarCameraFusionDetector::findNearestProjectedCloudPoint(const UavDetectorContext& context,
                                                               geometry_msgs::Point& point,
                                                               std::string& frame_id) const
{
  if (!context.image_detection.valid || !context.camera_model_ready() || !context.tf_buffer || !context.track_cloud ||
      context.image_header.frame_id.empty() || context.track_cloud->header.frame_id.empty() ||
      context.track_cloud->width * context.track_cloud->height == 0)
  {
    return false;
  }

  const auto& cloud = context.track_cloud;
  geometry_msgs::TransformStamped tf_msg;
  if (!context.getTransform(cloud->header.frame_id,
                            cloud->header.stamp,
                            tf_msg))
  {
    return false;
  }

  const cv::Point2f target_pixel(static_cast<float>(context.image_detection.pixel.x),
                                 static_cast<float>(context.image_detection.pixel.y));
  const double max_dist_sq = context.point_match_max_pixel_distance * context.point_match_max_pixel_distance;
  std::vector<cv::Point3f> object_points;
  std::vector<geometry_msgs::Point> source_points;
  object_points.reserve(cloud->width * cloud->height);
  source_points.reserve(cloud->width * cloud->height);

  try
  {
    // Transform valid cloud points first, then project them in one OpenCV batch.
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud, "z");
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
    {
      if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z))
      {
        continue;
      }

      geometry_msgs::PointStamped src;
      src.header = cloud->header;
      src.point.x = *iter_x;
      src.point.y = *iter_y;
      src.point.z = *iter_z;

      geometry_msgs::PointStamped dst;
      tf2::doTransform(src, dst, tf_msg);
      if (dst.point.z <= 1e-4)
      {
        continue;
      }

      object_points.emplace_back(static_cast<float>(dst.point.x),
                                 static_cast<float>(dst.point.y),
                                 static_cast<float>(dst.point.z));
      source_points.emplace_back(src.point);
    }
  }
  catch (const std::runtime_error& ex)
  {
    if (context.debug_mode)
    {
      ROS_WARN_THROTTLE(1.0, "[rm_radar_uav_detector][fusion] invalid PointCloud2 fields: %s", ex.what());
    }
    return false;
  }

  if (object_points.empty())
  {
    return false;
  }

  std::vector<cv::Point2f> projected_points;
  cv::projectPoints(object_points,
                    cv::Vec3d::all(0.0),
                    cv::Vec3d::all(0.0),
                    context.intrinsics,
                    context.dist_coeffs,
                    projected_points);
  if (projected_points.size() != source_points.size())
  {
    return false;
  }

  double best_dist_sq = std::numeric_limits<double>::max();
  geometry_msgs::Point best_point;
  // Linear nearest-neighbor search in image space; building a tree is not worth it for one query.
  for (std::size_t i = 0; i < projected_points.size(); ++i)
  {
    cv::Point2f projected_point = projected_points[i];
    if (!std::isfinite(projected_point.x) || !std::isfinite(projected_point.y))
    {
      continue;
    }

    projected_point.x += static_cast<float>(context.pixel_bias_u);
    projected_point.y += static_cast<float>(context.pixel_bias_v);
    if (projected_point.x < 0.0f || projected_point.x >= static_cast<float>(context.image_width) ||
        projected_point.y < 0.0f || projected_point.y >= static_cast<float>(context.image_height))
    {
      continue;
    }

    const double du = static_cast<double>(projected_point.x - target_pixel.x);
    const double dv = static_cast<double>(projected_point.y - target_pixel.y);
    const double dist_sq = du * du + dv * dv;
    if (dist_sq < best_dist_sq)
    {
      best_dist_sq = dist_sq;
      best_point = source_points[i];
    }
  }

  if (best_dist_sq > max_dist_sq)
  {
    return false;
  }

  point = best_point;
  frame_id = cloud->header.frame_id;
  return true;
}

void LidarCameraFusionDetector::updateMarkedKeypointOffset(const UavDetectorContext& context,
                                                           const geometry_msgs::Point& matched_point,
                                                           const std::string& frame_id)
{
  const auto& track = context.track;
  if (!context.has_track || !track.tracking || track.header.frame_id.empty() || frame_id.empty() ||
      frame_id != track.header.frame_id)
  {
    return;
  }

  std::lock_guard<std::mutex> lock(marked_keypoint_mutex_);
  // Store the selected point relative to the track centroid for later lidar-only frames.
  marked_keypoint_offset_state_.valid = true;
  marked_keypoint_offset_state_.offset_from_centroid.x = matched_point.x - track.position.x;
  marked_keypoint_offset_state_.offset_from_centroid.y = matched_point.y - track.position.y;
  marked_keypoint_offset_state_.offset_from_centroid.z = matched_point.z - track.position.z;
  marked_keypoint_offset_state_.frame_id = frame_id;
  marked_keypoint_offset_state_.stamp = context.image_header.stamp;
}

UavDetectorOutput LidarCameraFusionDetector::trackMarkedKeypoint(const UavDetectorContext& context) const
{
  UavDetectorOutput output;
  const auto& track = context.track;
  MarkedKeypointOffsetState state;
  if (!context.has_track || !track.tracking || track.header.frame_id.empty() || !getMarkedKeypointOffsetState(state) ||
      state.frame_id != track.header.frame_id)
  {
    return output;
  }

  const geometry_msgs::Point keypoint = markedKeypoint(state, track);
  cv::Point2f keypoint_pixel;
  // Reuse the stored offset to keep publishing a 2D point when camera detection is missing.
  if (!projectPointToImage(context, keypoint, track.header.frame_id, keypoint_pixel))
  {
    return output;
  }

  output.msg = makeTrackOutput(track, context.image_header.stamp);
  output.msg.position = keypoint;
  output.msg.header.frame_id = track.header.frame_id;
  output.msg.has_2d = true;
  output.msg.pixel.x = keypoint_pixel.x;
  output.msg.pixel.y = keypoint_pixel.y;
  output.msg.pixel.z = 0.0;
  output.has_projected_image_point = true;
  output.projected_image_point = keypoint_pixel;
  return output;
}

bool LidarCameraFusionDetector::getMarkedKeypointOffsetState(MarkedKeypointOffsetState& state) const
{
  std::lock_guard<std::mutex> lock(marked_keypoint_mutex_);
  state = marked_keypoint_offset_state_;
  return state.valid;
}

geometry_msgs::Point LidarCameraFusionDetector::markedKeypoint(const MarkedKeypointOffsetState& state,
                                                               const rm_radar_msgs::DroneTrackData& track)
{
  geometry_msgs::Point point;
  point.x = track.position.x + state.offset_from_centroid.x;
  point.y = track.position.y + state.offset_from_centroid.y;
  point.z = track.position.z + state.offset_from_centroid.z;
  return point;
}

rm_radar_msgs::UavDetection LidarCameraFusionDetector::makeTrackOutput(const rm_radar_msgs::DroneTrackData& track,
                                                                       const ros::Time& stamp)
{
  rm_radar_msgs::UavDetection msg;
  msg.header.stamp = stamp.isZero() ? track.header.stamp : stamp;
  msg.header.frame_id = track.header.frame_id;
  msg.valid = track.tracking;
  msg.confidence = 1.0;
  msg.has_vision = false;
  msg.error_x = 0.0;
  msg.error_y = 0.0;
  msg.has_3d = track.tracking;
  msg.tracking = track.tracking;
  msg.position = track.position;
  return msg;
}

bool LidarCameraFusionDetector::projectPointToImage(const UavDetectorContext& context,
                                                    const geometry_msgs::Point& point,
                                                    const std::string& frame_id,
                                                    cv::Point2f& pixel) const
{
  if (!context.camera_model_ready() || !context.tf_buffer || context.image_header.frame_id.empty() || frame_id.empty())
  {
    return false;
  }

  ros::Time source_stamp;
  // Pick the best available source stamp for TF lookup.
  if (context.track_cloud && frame_id == context.track_cloud->header.frame_id)
  {
    source_stamp = context.track_cloud->header.stamp;
  }
  else if (frame_id == context.track.header.frame_id)
  {
    source_stamp = context.track.header.stamp;
  }

  geometry_msgs::TransformStamped tf_msg;
  if (!context.getTransform(frame_id,
                            source_stamp,
                            tf_msg))
  {
    return false;
  }

  geometry_msgs::PointStamped src;
  src.header.frame_id = frame_id;
  src.header.stamp = source_stamp;
  src.point = point;
  geometry_msgs::PointStamped dst;
  tf2::doTransform(src, dst, tf_msg);
  if (dst.point.z <= 1e-4)
  {
    return false;
  }

  std::vector<cv::Point3f> object_points(1);
  object_points[0] = cv::Point3f(static_cast<float>(dst.point.x),
                                 static_cast<float>(dst.point.y),
                                 static_cast<float>(dst.point.z));
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
    return false;
  }

  pixel = projected_points.front();
  pixel.x += static_cast<float>(context.pixel_bias_u);
  pixel.y += static_cast<float>(context.pixel_bias_v);
  return pixel.x >= 0.0f && pixel.x < static_cast<float>(context.image_width) && pixel.y >= 0.0f &&
         pixel.y < static_cast<float>(context.image_height);
}

}  // namespace rm_radarplugin
