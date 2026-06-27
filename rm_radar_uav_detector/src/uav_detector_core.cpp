#include "uav_detector_core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <boost/make_shared.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <pluginlib/class_list_macros.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

PLUGINLIB_EXPORT_CLASS(rm_radarplugin::UavDetectorCore, nodelet::Nodelet)

namespace rm_radarplugin
{

void UavDetectorCore::onInit()
{
  nh_ = getNodeHandle();
  pnh_ = getPrivateNodeHandle();

  pnh_.param<std::string>("image_topic", image_topic_, "/hk_camera_right/image_raw");
  pnh_.param<std::string>("track_topic", track_topic_, "/lidar_track");
  pnh_.param<std::string>("image_detection_topic", image_detection_topic_, "/right_camera_Processor/processor/single_result_msg");
  pnh_.param<std::string>("roi_topic", roi_topic_, "/rm_radar_uav_detector/roi_image");
  pnh_.param<std::string>("debug_topic", debug_topic_, "/rm_radar_uav_detector/debug_image");
  pnh_.param<std::string>("uav_detection_topic", uav_detection_topic_, "/rm_radar_uav_detector/uav_detection");
  pnh_.param<std::string>("fixed_frame", fixed_frame_, "base_link");
  pnh_.param<bool>("debug_mode", debug_mode_, debug_mode_);
  pnh_.param<double>("timeout_s", timeout_s_, timeout_s_);
  pnh_.param<double>("tf_timeout_s", tf_timeout_s_, tf_timeout_s_);
  pnh_.param<double>("image_detection_timeout_s", image_detection_timeout_s_, image_detection_timeout_s_);
  pnh_.param<double>("point_match_max_pixel_distance", point_match_max_pixel_distance_, point_match_max_pixel_distance_);
  pnh_.param<int>("line_thickness", line_thickness_, line_thickness_);
  pnh_.param<bool>("disable", disable_, disable_);
  pnh_.param<bool>("draw_bbox", draw_bbox_, draw_bbox_);
  pnh_.param<bool>("draw_roi", draw_roi_, draw_roi_);
  pnh_.param<double>("pixel_bias_u", pixel_bias_u_, pixel_bias_u_);
  pnh_.param<double>("pixel_bias_v", pixel_bias_v_, pixel_bias_v_);
  pnh_.param<bool>("need_roi", need_roi_, need_roi_);
  pnh_.param<float>("expand_ratio", expand_ratio_, expand_ratio_);

  // Clamp parameters that would otherwise make synchronization or drawing invalid.
  timeout_s_ = std::max(0.01, timeout_s_);
  tf_timeout_s_ = std::max(0.001, tf_timeout_s_);
  image_detection_timeout_s_ = std::max(0.01, image_detection_timeout_s_);
  point_match_max_pixel_distance_ = std::max(0.0, point_match_max_pixel_distance_);
  line_thickness_ = std::max(1, line_thickness_);
  expand_ratio_ = std::max(0.0f, expand_ratio_);

  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(tf_buffer_);
  it_ = std::make_shared<image_transport::ImageTransport>(nh_);
  image_sub_ = it_->subscribeCamera(image_topic_, 1, &UavDetectorCore::imageCallback, this);
  debug_image_pub_ = it_->advertise(debug_topic_, 1);
  roi_image_pub_ = it_->advertise(roi_topic_, 1);
  uav_detection_pub_ = nh_.advertise<rm_radar_msgs::UavDetection>(uav_detection_topic_, 1);
  track_sub_ = nh_.subscribe(track_topic_, 10, &UavDetectorCore::lidarDetectionCallback, this);
  image_detection_sub_ = nh_.subscribe(image_detection_topic_, 10, &UavDetectorCore::imageDetectionCallback, this);

  cfg_server_ = std::make_unique<dynamic_reconfigure::Server<rm_radar_uav_detector::UavDetectorConfig>>(pnh_);
  cfg_cb_ = [this](rm_radar_uav_detector::UavDetectorConfig& config, uint32_t level) {
    reconfigCB(config, level);
  };
  cfg_server_->setCallback(cfg_cb_);

  rm_radar_uav_detector::UavDetectorConfig cfg_init;
  cfg_init.debug_mode = debug_mode_;
  cfg_init.timeout_s = timeout_s_;
  cfg_init.tf_timeout_s = tf_timeout_s_;
  cfg_init.disable = disable_;
  cfg_init.draw_bbox = draw_bbox_;
  cfg_init.draw_roi = draw_roi_;
  cfg_init.line_thickness = line_thickness_;
  cfg_init.expand_ratio = expand_ratio_;
  cfg_init.pixel_bias_u = pixel_bias_u_;
  cfg_init.pixel_bias_v = pixel_bias_v_;
  cfg_server_->updateConfig(cfg_init);

  NODELET_INFO("[rm_radar_uav_detector] inited image_topic=%s track_topic=%s image_detection_topic=%s debug_topic=%s",
               image_topic_.c_str(),
               track_topic_.c_str(),
               image_detection_topic_.c_str(),
               debug_topic_.c_str());
}

void UavDetectorCore::lidarDetectionCallback(const rm_radar_msgs::DroneTrackData::ConstPtr& msg)
{
  if (!msg)
  {
    return;
  }

  std::lock_guard<std::mutex> lock(track_mutex_);
  // Keep a short history so each image can choose the closest lidar track by stamp.
  track_buffer_.push_back(*msg);
  while (track_buffer_.size() > kTrackBufferMaxSize)
  {
    track_buffer_.pop_front();
  }
}

void UavDetectorCore::imageDetectionCallback(const rm_radar_msgs::DroneDetection::ConstPtr& msg)
{
  if (!msg)
  {
    camera_detector_.clear();
    return;
  }

  camera_detector_.updateDetection(*msg);
}

void UavDetectorCore::imageCallback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info)
{
  if (!img)
  {
    return;
  }

  updateCameraModel(info);

  cv_bridge::CvImageConstPtr cv_image;
  try
  {
    cv_image = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
  }
  catch (const cv_bridge::Exception& ex)
  {
    NODELET_WARN_THROTTLE(1.0, "[rm_radar_uav_detector] image conversion failed: %s", ex.what());
    return;
  }

  const RuntimeParams params = getRuntimeParams();

  // Build one immutable snapshot for all detectors used by this image.
  rm_radar_msgs::DroneTrackData nearest_track;
  const bool has_track = getNearestTrack(img->header.stamp, nearest_track);
  const UavDetectorContext context = buildContext(img->header, params, has_track, nearest_track);

  const UavDetectorOutput camera_output = camera_detector_.detect(context);

  UavDetectorOutput selected_output;
  const Bbox2D::Bbox* debug_bbox = nullptr;
  std::string status_text;

  if (hasValidCameraDetection(context))
  {
    // Vision has priority; lidar is used to recover 3D position and bbox when possible.
    selected_output = camera_output;
    status_text = "camera";
    geometry_msgs::Point matched_point;
    std::string matched_frame_id;
    if (lidar_camera_fusion_detector_.findNearestProjectedCloudPoint(context, matched_point, matched_frame_id))
    {
      selected_output.msg.has_3d = true;
      selected_output.msg.tracking = false;
      selected_output.msg.position = matched_point;
      selected_output.msg.header.frame_id = matched_frame_id;
      status_text = "camera_cloud";
      if (has_track && nearest_track.tracking)
      {
        lidar_camera_fusion_detector_.updateMarkedKeypointOffset(context, matched_point, matched_frame_id);
      }
    }

    if (has_track && context.camera_model_ready())
    {
      const UavDetectorOutput fusion_output = lidar_camera_fusion_detector_.detect(context);
      if (fusion_output.msg.valid)
      {
        selected_output.msg.tracking = fusion_output.msg.tracking;
        if (!selected_output.msg.has_3d)
        {
          selected_output.msg.header.frame_id = fusion_output.msg.header.frame_id;
          selected_output.msg.has_3d = fusion_output.msg.has_3d;
          selected_output.msg.position = fusion_output.msg.position;
        }
        status_text = selected_output.msg.has_3d ? "camera_cloud_track" : "camera_track";
      }

      if (fusion_output.has_bbox)
      {
        selected_output.has_bbox = true;
        selected_output.bbox = fusion_output.bbox;
        debug_bbox = &selected_output.bbox;
        {
          std::lock_guard<std::mutex> lock(bbox_mutex_);
          drone_bbox_.bbox2d = fusion_output.bbox;
        }
        publishRoiImage(cv_image, fusion_output.bbox, params.debug_mode);
      }
      else
      {
        clearDroneBbox();
      }
    }
    else
    {
      clearDroneBbox();
    }
  }
  else if (has_track)
  {
    // Without a fresh vision result, fall back to lidar track and the last marked keypoint.
    UavDetectorOutput fusion_output;
    UavDetectorOutput tracked_keypoint_output = lidar_camera_fusion_detector_.trackMarkedKeypoint(context);
    const bool has_tracked_keypoint = tracked_keypoint_output.msg.valid && tracked_keypoint_output.msg.has_2d;
    if (context.camera_model_ready())
    {
      fusion_output = lidar_camera_fusion_detector_.detect(context);
      tracked_keypoint_output.has_bbox = fusion_output.has_bbox;
      tracked_keypoint_output.bbox = fusion_output.bbox;
    }

    if (fusion_output.has_bbox)
    {
      selected_output = has_tracked_keypoint ? tracked_keypoint_output : fusion_output;
      selected_output.has_bbox = fusion_output.has_bbox;
      selected_output.bbox = fusion_output.bbox;
      debug_bbox = &selected_output.bbox;
      status_text = has_tracked_keypoint ? "tracked_keypoint" : "fusion";
      {
        std::lock_guard<std::mutex> lock(bbox_mutex_);
        drone_bbox_.bbox2d = fusion_output.bbox;
      }
      publishRoiImage(cv_image, fusion_output.bbox, params.debug_mode);
    }
    else if (has_tracked_keypoint)
    {
      selected_output = tracked_keypoint_output;
      status_text = "tracked_keypoint";
      clearDroneBbox();
    }
    else if (fusion_output.msg.valid)
    {
      selected_output = fusion_output;
      status_text = "track_only";
      clearDroneBbox();
    }
    else
    {
      status_text = context.camera_model_ready() ? "no_projection" : "no_camera_model";
      clearDroneBbox();
    }
  }
  else
  {
    status_text = "no_detection";
    clearDroneBbox();
  }

  updateProjectedImagePoint(selected_output);
  publishOutput(selected_output);
  drawBbox(cv_image, debug_bbox, status_text);
}

void UavDetectorCore::reconfigCB(rm_radar_uav_detector::UavDetectorConfig& config, uint32_t /*level*/)
{
  std::lock_guard<std::mutex> lock(param_mutex_);
  debug_mode_ = config.debug_mode;
  timeout_s_ = std::max(0.01, config.timeout_s);
  tf_timeout_s_ = std::max(0.001, config.tf_timeout_s);
  disable_ = config.disable;
  draw_bbox_ = config.draw_bbox;
  draw_roi_ = config.draw_roi;
  line_thickness_ = std::max(1, config.line_thickness);
  expand_ratio_ = std::max(0.0f, static_cast<float>(config.expand_ratio));
  pixel_bias_u_ = config.pixel_bias_u;
  pixel_bias_v_ = config.pixel_bias_v;
}

void UavDetectorCore::updateCameraModel(const sensor_msgs::CameraInfoConstPtr& info)
{
  if (!info)
  {
    return;
  }

  // Convert CameraInfo into OpenCV matrices used by cv::projectPoints.
  cv::Mat intrinsics(3, 3, CV_64F);
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      intrinsics.at<double>(row, col) = info->K[row * 3 + col];
    }
  }

  cv::Mat dist_coeffs(1, static_cast<int>(info->D.size()), CV_64F);
  for (std::size_t i = 0; i < info->D.size(); ++i)
  {
    dist_coeffs.at<double>(0, static_cast<int>(i)) = info->D[i];
  }

  std::lock_guard<std::mutex> lock(camera_model_mutex_);
  intrinsics_ = intrinsics;
  dist_coeffs_ = dist_coeffs;
  camera_info_ = info;
  camera_model_initialized_ = true;
  image_width_ = static_cast<int>(info->width);
  image_height_ = static_cast<int>(info->height);
}

bool UavDetectorCore::getNearestTrack(const ros::Time& image_stamp, rm_radar_msgs::DroneTrackData& track) const
{
  double timeout_s = 0.2;
  {
    std::lock_guard<std::mutex> lock(param_mutex_);
    timeout_s = timeout_s_;
  }

  std::lock_guard<std::mutex> lock(track_mutex_);
  if (track_buffer_.empty())
  {
    return false;
  }

  if (image_stamp.isZero())
  {
    // Untimed images cannot be synchronized, so use the newest track sample.
    track = track_buffer_.back();
    return track.tracking;
  }

  auto best_it = track_buffer_.end();
  double best_dt = std::numeric_limits<double>::max();
  // Prefer the track closest to the image time, bounded by timeout_s_.
  for (auto it = track_buffer_.begin(); it != track_buffer_.end(); ++it)
  {
    if (it->header.stamp.isZero())
    {
      continue;
    }

    const double dt = std::abs((image_stamp - it->header.stamp).toSec());
    if (dt < best_dt)
    {
      best_dt = dt;
      best_it = it;
    }
  }

  if (best_it == track_buffer_.end() || best_dt > timeout_s)
  {
    return false;
  }

  track = *best_it;
  return track.tracking;
}

UavDetectorContext UavDetectorCore::buildContext(const std_msgs::Header& image_header,
                                                 const RuntimeParams& params,
                                                 bool has_track,
                                                 const rm_radar_msgs::DroneTrackData& track) const
{
  UavDetectorContext context;
  context.image_header = image_header;
  context.debug_mode = params.debug_mode;
  context.need_roi = params.need_roi;
  context.expand_ratio = params.expand_ratio;
  context.tf_timeout_s = params.tf_timeout_s;
  context.image_detection_timeout_s = params.image_detection_timeout_s;
  context.pixel_bias_u = params.pixel_bias_u;
  context.pixel_bias_v = params.pixel_bias_v;
  context.fixed_frame = params.fixed_frame;
  context.tf_buffer = const_cast<tf2_ros::Buffer*>(&tf_buffer_);
  context.point_match_max_pixel_distance = params.point_match_max_pixel_distance;
  context.image_detection = camera_detector_.getLatestDetection();
  context.has_track = has_track;
  context.track = track;

  if (has_track && track.track_cloud.width * track.track_cloud.height > 0)
  {
    // Share a const copy of the embedded cloud so fusion can project individual points.
    context.track_cloud = boost::make_shared<sensor_msgs::PointCloud2 const>(track.track_cloud);
  }

  {
    std::lock_guard<std::mutex> lock(camera_model_mutex_);
    if (camera_model_initialized_)
    {
      // Clone matrices so downstream detectors do not hold the camera_model_mutex_.
      context.intrinsics = intrinsics_.clone();
      context.dist_coeffs = dist_coeffs_.clone();
      context.image_width = image_width_;
      context.image_height = image_height_;
    }
  }

  return context;
}

bool UavDetectorCore::hasValidCameraDetection(const UavDetectorContext& context)
{
  const ImageDetection& detection = context.image_detection;
  if (!detection.valid)
  {
    return false;
  }

  if (context.image_header.stamp.isZero() || detection.header.stamp.isZero())
  {
    return true;
  }

  const double age_s = std::abs((context.image_header.stamp - detection.header.stamp).toSec());
  return age_s <= context.image_detection_timeout_s;
}

void UavDetectorCore::publishRoiImage(const cv_bridge::CvImageConstPtr& image,
                                      const Bbox2D::Bbox& bbox,
                                      bool debug_mode)
{
  if (!image || image->image.empty() || !bbox.need_roi || bbox.roi_area.area() <= 0)
  {
    return;
  }

  const cv::Rect image_rect(0, 0, image->image.cols, image->image.rows);
  const cv::Rect roi_area = bbox.roi_area & image_rect;
  if (roi_area.area() <= 0)
  {
    if (debug_mode)
    {
      NODELET_WARN_THROTTLE(1.0, "[rm_radar_uav_detector] ROI outside image.");
    }
    return;
  }

  roi_image_pub_.publish(cv_bridge::CvImage(image->header,
                                            sensor_msgs::image_encodings::BGR8,
                                            image->image(roi_area).clone())
                             .toImageMsg());
}

void UavDetectorCore::drawBbox(const cv_bridge::CvImageConstPtr& temp,
                               const Bbox2D::Bbox* bbox,
                               const std::string& status_text)
{
  if (!temp || temp->image.empty())
  {
    return;
  }

  bool disable = false;
  bool draw_bbox = false;
  bool draw_roi = false;
  int line_thickness = 1;
  {
    std::lock_guard<std::mutex> lock(param_mutex_);
    disable = disable_;
    draw_bbox = draw_bbox_;
    draw_roi = draw_roi_;
    line_thickness = line_thickness_;
  }

  if (disable)
  {
    return;
  }

  cv::Mat debug_image = temp->image.clone();
  const cv::Scalar bbox_color(0.0, 255.0, 0.0);
  const cv::Scalar roi_color(0.0, 0.0, 255.0);
  const cv::Scalar projected_point_color(255.0, 0.0, 255.0);

  const bool has_valid_bbox = (bbox != nullptr && bbox->is_valid && bbox->bbox_area.area() > 0);
  if (has_valid_bbox && draw_bbox)
  {
    cv::rectangle(debug_image, bbox->bbox_area, bbox_color, line_thickness);
    cv::putText(debug_image,
                "bbox",
                cv::Point(bbox->bbox_area.x, std::max(0, bbox->bbox_area.y - 6)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                bbox_color,
                std::max(1, line_thickness));

    const cv::Point min_pt(cvRound(bbox->bbox_min_pt.x), cvRound(bbox->bbox_min_pt.y));
    const cv::Point max_pt(cvRound(bbox->bbox_max_pt.x), cvRound(bbox->bbox_max_pt.y));
    const cv::Point target_pt(cvRound(bbox->target_pt.x), cvRound(bbox->target_pt.y));
    const int point_radius = std::max(2, line_thickness + 1);
    cv::circle(debug_image, min_pt, point_radius, cv::Scalar(255.0, 255.0, 0.0), -1);
    cv::circle(debug_image, max_pt, point_radius, cv::Scalar(255.0, 255.0, 0.0), -1);
    cv::circle(debug_image, target_pt, point_radius, cv::Scalar(0.0, 255.0, 255.0), -1);
    cv::putText(debug_image, "min", min_pt + cv::Point(4, -4), cv::FONT_HERSHEY_SIMPLEX, 0.45, bbox_color, 1);
    cv::putText(debug_image, "max", max_pt + cv::Point(4, -4), cv::FONT_HERSHEY_SIMPLEX, 0.45, bbox_color, 1);
    cv::putText(debug_image, "target", target_pt + cv::Point(4, -4), cv::FONT_HERSHEY_SIMPLEX, 0.45, bbox_color, 1);
  }

  if (has_valid_bbox && draw_roi && bbox->need_roi && bbox->roi_area.area() > 0)
  {
    cv::rectangle(debug_image, bbox->roi_area, roi_color, line_thickness);
    cv::putText(debug_image,
                "roi",
                cv::Point(bbox->roi_area.x, std::max(0, bbox->roi_area.y - 6)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                roi_color,
                std::max(1, line_thickness));
  }

  cv::Point2f projected_image_point;
  bool projected_image_point_valid = false;
  {
    std::lock_guard<std::mutex> lock(projected_image_point_mutex_);
    projected_image_point = projected_image_point_;
    projected_image_point_valid = projected_image_point_valid_;
  }
  if (projected_image_point_valid)
  {
    const cv::Point pt(cvRound(projected_image_point.x), cvRound(projected_image_point.y));
    const int cross_size = std::max(8, line_thickness * 4);
    cv::line(debug_image,
             pt + cv::Point(-cross_size, 0),
             pt + cv::Point(cross_size, 0),
             projected_point_color,
             line_thickness);
    cv::line(debug_image,
             pt + cv::Point(0, -cross_size),
             pt + cv::Point(0, cross_size),
             projected_point_color,
             line_thickness);
    cv::putText(debug_image,
                "image_detection",
                pt + cv::Point(4, -4),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                projected_point_color,
                std::max(1, line_thickness));
  }

  if (!status_text.empty())
  {
    cv::putText(debug_image,
                status_text,
                cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0.0, 0.0, 255.0),
                2);
  }

  debug_image_pub_.publish(cv_bridge::CvImage(temp->header, sensor_msgs::image_encodings::BGR8, debug_image).toImageMsg());
}

}  // namespace rm_radarplugin
