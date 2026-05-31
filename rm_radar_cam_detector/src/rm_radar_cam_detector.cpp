#include "rm_radar_cam_detector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <pluginlib/class_list_macros.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

PLUGINLIB_EXPORT_CLASS(rm_radarplugin::CamDetector, nodelet::Nodelet)

namespace rm_radarplugin
{
  void CamDetector::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();

    pnh_.param<std::string>("image_topic", image_topic_, "/left_camera/image_raw");
    pnh_.param<std::string>("track_topic", track_topic_, "/lidar_track");
    pnh_.param<std::string>("roi_topic", roi_topic_, "/rm_radar_cam_detector/roi_image");
    pnh_.param<std::string>("debug_topic", debug_topic_, "/rm_radar_cam_detector/debug_image");
    pnh_.param<std::string>("fixed_frame", fixed_frame_, fixed_frame_);
    pnh_.param<bool>("debug_mode", debug_mode_, debug_mode_);
    pnh_.param<double>("timeout_s", timeout_s_, timeout_s_);
    pnh_.param<double>("tf_timeout_s", tf_timeout_s_, tf_timeout_s_);
    pnh_.param<int>("line_thickness", line_thickness_, line_thickness_);
    pnh_.param<bool>("disable", disable_, disable_);
    pnh_.param<bool>("draw_bbox", draw_bbox_, draw_bbox_);
    pnh_.param<bool>("draw_roi", draw_roi_, draw_roi_);
    pnh_.param<double>("pixel_bias_u", pixel_bias_u_, pixel_bias_u_);
    pnh_.param<double>("pixel_bias_v", pixel_bias_v_, pixel_bias_v_);
    if (fixed_frame_.empty())
    {
      fixed_frame_ = "base_link";
    }

    pnh_.param<bool>("need_roi", need_roi_, need_roi_);
    pnh_.param<float>("expand_ratio", expand_ratio_, expand_ratio_);
    expand_ratio_ = std::max(0.0f, expand_ratio_);

    line_thickness_ = std::max(1, line_thickness_);

    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(tf_buffer_);
    it_ = std::make_shared<image_transport::ImageTransport>(nh_);
    image_sub_ = it_->subscribeCamera(image_topic_, 1, &CamDetector::cam_callback, this);
    debug_image_pub_ = it_->advertise(debug_topic_, 1);
    roi_image_pub_ = it_->advertise(roi_topic_, 1);
    track_sub_ = nh_.subscribe(track_topic_, 10, &CamDetector::trackCallback, this);

    cfg_server_ = std::make_unique<dynamic_reconfigure::Server<rm_radar_cam_detector::CamDetectorConfig>>(pnh_);
    cfg_cb_ = [this](rm_radar_cam_detector::CamDetectorConfig& config, uint32_t level) {
      reconfigCB(config, level);
    };
    cfg_server_->setCallback(cfg_cb_);
    rm_radar_cam_detector::CamDetectorConfig cfg_init;
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

    NODELET_INFO("[rm_radar_cam_detector] inited image_topic=%s track_topic=%s debug_topic=%s",
                image_topic_.c_str(), track_topic_.c_str(), debug_topic_.c_str());
  }

  void CamDetector::reconfigCB(rm_radar_cam_detector::CamDetectorConfig& config, uint32_t /*level*/)
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

  bool CamDetector::isValid(const std::vector<cv::Point2f>& projected_points,
                            const cv::Point2f& projected_target_point,
                            bool debug_mode,
                            bool need_roi,
                            float expand_ratio,
                            int image_width,
                            int image_height)
  {
    std::lock_guard<std::mutex> lock(bbox_mutex_);
    drone_bbox_.clear();
    if (projected_points.empty())
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Projected points empty.");
      }
      return false;
    }

    float bbox_min_x = std::numeric_limits<float>::max();
    float bbox_min_y = std::numeric_limits<float>::max();
    float bbox_max_x = std::numeric_limits<float>::lowest();
    float bbox_max_y = std::numeric_limits<float>::lowest();

    for (const auto& p : projected_points)
    {
      if (!std::isfinite(p.x) || !std::isfinite(p.y))
      {
        if (debug_mode)
        {
          ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Projected point is NaN/Inf.");
        }
        return false;
      }
      bbox_min_x = std::min(bbox_min_x, p.x);
      bbox_min_y = std::min(bbox_min_y, p.y);
      bbox_max_x = std::max(bbox_max_x, p.x);
      bbox_max_y = std::max(bbox_max_y, p.y);
    }

    if (bbox_min_x > bbox_max_x || bbox_min_y > bbox_max_y)
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Invalid bbox min/max after projection.");
      }
      return false;
    }

    const float min_x = 0.0f;
    const float min_y = 0.0f;
    const float max_x = static_cast<float>(image_width - 1);
    const float max_y = static_cast<float>(image_height - 1);

    const bool bbox_outside =
        (bbox_min_x < min_x || bbox_max_x > max_x || bbox_min_y < min_y || bbox_max_y > max_y);
    float clipped_min_x = std::max(min_x, std::min(bbox_min_x, max_x));
    float clipped_min_y = std::max(min_y, std::min(bbox_min_y, max_y));
    float clipped_max_x = std::max(min_x, std::min(bbox_max_x, max_x));
    float clipped_max_y = std::max(min_y, std::min(bbox_max_y, max_y));
    if (clipped_max_x < clipped_min_x || clipped_max_y < clipped_min_y)
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Bbox has no overlap with image.");
      }
      return false;
    }

    const float raw_w = std::max(1e-3f, bbox_max_x - bbox_min_x);
    const float raw_h = std::max(1e-3f, bbox_max_y - bbox_min_y);
    const float clipped_w = std::max(0.0f, clipped_max_x - clipped_min_x);
    const float clipped_h = std::max(0.0f, clipped_max_y - clipped_min_y);
    const double raw_area = static_cast<double>(raw_w) * static_cast<double>(raw_h);
    const double clipped_area = static_cast<double>(clipped_w) * static_cast<double>(clipped_h);
    const double visible_ratio = (raw_area > 1e-9) ? (clipped_area / raw_area) : 0.0;

    if (bbox_outside)
    {
      if (visible_ratio < 0.02)
      {
        if (debug_mode)
        {
          ROS_WARN_THROTTLE(
              1.0,
              "[rm_radar_cam_detector][4] Partial bbox visible ratio too low (ratio=%.3f, min=%.3f).",
              visible_ratio,
              0.02);
        }
        return false;
      }
      bbox_min_x = clipped_min_x;
      bbox_min_y = clipped_min_y;
      bbox_max_x = clipped_max_x;
      bbox_max_y = clipped_max_y;
    }

    auto& bbox = drone_bbox_.bbox2d;
    bbox.bbox_min_pt = cv::Point2f(bbox_min_x, bbox_min_y);
    bbox.bbox_max_pt = cv::Point2f(bbox_max_x, bbox_max_y);
    bbox.target_pt = projected_target_point;
    const int bbox_x0 = static_cast<int>(std::floor(bbox_min_x));
    const int bbox_y0 = static_cast<int>(std::floor(bbox_min_y));
    const int bbox_x1 = static_cast<int>(std::ceil(bbox_max_x));
    const int bbox_y1 = static_cast<int>(std::ceil(bbox_max_y));
    bbox.bbox_area = cv::Rect(
        bbox_x0,
        bbox_y0,
        std::max(1, bbox_x1 - bbox_x0 + 1),
        std::max(1, bbox_y1 - bbox_y0 + 1));
    bbox.is_valid = true;
    bbox.last_update_time = ros::Time::now();

    if (need_roi)
    {
      // Expand ROI around bbox center using expand ratio (0.0 means no expansion).
      const float bbox_cx = 0.5f * (bbox_min_x + bbox_max_x);
      const float bbox_cy = 0.5f * (bbox_min_y + bbox_max_y);
      const float bbox_half_w = 0.5f * (bbox_max_x - bbox_min_x);
      const float bbox_half_h = 0.5f * (bbox_max_y - bbox_min_y);
      const float ratio = std::max(0.0f, expand_ratio);
      const float roi_half_w = bbox_half_w * (1.0f + ratio);
      const float roi_half_h = bbox_half_h * (1.0f + ratio);

      const float roi_min_x = bbox_cx - roi_half_w;
      const float roi_max_x = bbox_cx + roi_half_w;
      const float roi_min_y = bbox_cy - roi_half_h;
      const float roi_max_y = bbox_cy + roi_half_h;

      const int roi_x0 = std::max(0, std::min(static_cast<int>(std::floor(roi_min_x)), image_width - 1));
      const int roi_y0 = std::max(0, std::min(static_cast<int>(std::floor(roi_min_y)), image_height - 1));
      const int roi_x1 = std::max(0, std::min(static_cast<int>(std::ceil(roi_max_x)), image_width - 1));
      const int roi_y1 = std::max(0, std::min(static_cast<int>(std::ceil(roi_max_y)), image_height - 1));
      if (roi_x1 >= roi_x0 && roi_y1 >= roi_y0)
      {
        bbox.roi_area = cv::Rect(roi_x0, roi_y0, roi_x1 - roi_x0 + 1, roi_y1 - roi_y0 + 1);
        bbox.need_roi = bbox.roi_area.area() > 0;
        bbox.roi_x = bbox.roi_area.x;
        bbox.roi_y = bbox.roi_area.y;
        bbox.expand_ratio = ratio;
      }
    }

    return true;
  }
  bool CamDetector::projectBboxToImage(const rm_radar_msgs::DroneTrackData& track,
                                       const ros::Time& image_stamp,
                                       const std::string& target_frame,
                                       double tf_timeout_s,
                                       bool need_roi,
                                       float expand_ratio,
                                       bool debug_mode,
                                       const cv::Mat& intrinsics,
                                       const cv::Mat& dist_coeffs,
                                       int image_width,
                                       int image_height)
  {
    if (track.header.frame_id.empty())
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Track frame_id is empty.");
      }
      return false;
    }

    geometry_msgs::TransformStamped tf_msg;
    try
    {
      if (!image_stamp.isZero() && !track.header.stamp.isZero())
      {
        tf_msg = tf_buffer_.lookupTransform(
            target_frame,
            image_stamp,
            track.header.frame_id,
            track.header.stamp,
            fixed_frame_,
            ros::Duration(tf_timeout_s));
      }
      else
      {
        tf_msg = tf_buffer_.lookupTransform(
            target_frame, track.header.frame_id, track.header.stamp, ros::Duration(tf_timeout_s));
      }
    }
    catch (const tf2::TransformException& ex)
    {
      if (debug_mode)
      {
        NODELET_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] TF lookup failed: %s", ex.what());
      }
      return false;
    }

    if (track.aabb_points.empty())
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Track aabb_points is empty.");
      }
      return false;
    }

    std::vector<cv::Point3f> points_cam;
    points_cam.reserve(track.aabb_points.size());
    for (const auto& pt : track.aabb_points)
    {
      geometry_msgs::PointStamped p_src;
      p_src.header = track.header;
      p_src.point = pt;

      geometry_msgs::PointStamped p_cam;
      tf2::doTransform(p_src, p_cam, tf_msg);
      if (p_cam.point.z <= 1e-4)
      {
        if (debug_mode)
        {
          ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Point behind camera or too close (z<=1e-4).");
        }
        return false;
      }
      points_cam.emplace_back(
          static_cast<float>(p_cam.point.x), static_cast<float>(p_cam.point.y), static_cast<float>(p_cam.point.z));
    }

    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(points_cam, cv::Vec3d::all(0.0), cv::Vec3d::all(0.0), intrinsics, dist_coeffs, projected_points);
    if (projected_points.size() != track.aabb_points.size())
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Projected point count mismatch.");
      }
      return false;
    }

    geometry_msgs::PointStamped target_src;
    target_src.header = track.header;
    target_src.point = track.position;

    geometry_msgs::PointStamped target_cam;
    tf2::doTransform(target_src, target_cam, tf_msg);
    if (target_cam.point.z <= 1e-4)
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Track centroid behind camera or too close (z<=1e-4).");
      }
      return false;
    }
    std::vector<cv::Point3f> target_point_cam(1);
    target_point_cam[0] =
        cv::Point3f(static_cast<float>(target_cam.point.x), static_cast<float>(target_cam.point.y), static_cast<float>(target_cam.point.z));
    std::vector<cv::Point2f> projected_target_points;
    cv::projectPoints(
        target_point_cam, cv::Vec3d::all(0.0), cv::Vec3d::all(0.0), intrinsics, dist_coeffs, projected_target_points);
    if (projected_target_points.size() != 1)
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Failed to project track centroid.");
      }
      return false;
    }
    cv::Point2f projected_target_point = projected_target_points.front();

    double pixel_bias_u = 0.0;
    double pixel_bias_v = 0.0;
    {
      std::lock_guard<std::mutex> lock(param_mutex_);
      pixel_bias_u = pixel_bias_u_;
      pixel_bias_v = pixel_bias_v_;
    }
    if (std::abs(pixel_bias_u) > 1e-9 || std::abs(pixel_bias_v) > 1e-9)
    {
      for (auto& p : projected_points)
      {
        p.x += static_cast<float>(pixel_bias_u);
        p.y += static_cast<float>(pixel_bias_v);
      }
      projected_target_point.x += static_cast<float>(pixel_bias_u);
      projected_target_point.y += static_cast<float>(pixel_bias_v);
      if (debug_mode)
      {
        ROS_INFO_THROTTLE(1.0,
                          "[rm_radar_cam_detector][4] Applied pixel bias du=%.2f dv=%.2f",
                          pixel_bias_u,
                          pixel_bias_v);
      }
    }

    if (!std::isfinite(projected_target_point.x) || !std::isfinite(projected_target_point.y))
    {
      if (debug_mode)
      {
        ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][4] Projected track centroid is NaN/Inf.");
      }
      return false;
    }
    return isValid(
        projected_points,
        projected_target_point,
        debug_mode,
        need_roi,
        expand_ratio,
        image_width,
        image_height);
  }

  void CamDetector::drawBbox(const cv_bridge::CvImageConstPtr& temp, const Bbox2D::Bbox* bbox, const std::string& status_text)
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

    cv::Mat debug_image = temp->image.clone();
    const cv::Scalar bbox_color(0.0, 255.0, 0.0);  // BGR green
    const cv::Scalar roi_color(0.0, 0.0, 255.0);   // BGR red

    const bool has_valid_bbox = (bbox != nullptr && bbox->is_valid && bbox->bbox_area.area() > 0);
    if (!disable && has_valid_bbox && draw_bbox)
    {
      cv::rectangle(debug_image, bbox->bbox_area, bbox_color, line_thickness);
      cv::putText(
          debug_image,
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
      cv::circle(debug_image, min_pt, point_radius, cv::Scalar(255.0, 255.0, 0.0), -1);      // yellow
      cv::circle(debug_image, max_pt, point_radius, cv::Scalar(255.0, 255.0, 0.0), -1);      // yellow
      cv::circle(debug_image, target_pt, point_radius, cv::Scalar(0.0, 255.0, 255.0), -1);   // orange
      cv::putText(debug_image, "min", min_pt + cv::Point(4, -4), cv::FONT_HERSHEY_SIMPLEX, 0.45, bbox_color, 1);
      cv::putText(debug_image, "max", max_pt + cv::Point(4, -4), cv::FONT_HERSHEY_SIMPLEX, 0.45, bbox_color, 1);
      cv::putText(debug_image, "target", target_pt + cv::Point(4, -4), cv::FONT_HERSHEY_SIMPLEX, 0.45, bbox_color, 1);
    }

    if (!disable && has_valid_bbox && draw_roi && bbox->need_roi && bbox->roi_area.area() > 0)
    {
      cv::rectangle(debug_image, bbox->roi_area, roi_color, line_thickness);
      cv::putText(
          debug_image,
          "roi",
          cv::Point(bbox->roi_area.x, std::max(0, bbox->roi_area.y - 6)),
          cv::FONT_HERSHEY_SIMPLEX,
          0.5,
          roi_color,
          std::max(1, line_thickness));
    }

    if (!status_text.empty())
    {
      cv::putText(
          debug_image,
          status_text,
          cv::Point(20, 40),
          cv::FONT_HERSHEY_SIMPLEX,
          1.0,
          cv::Scalar(0.0, 0.0, 255.0),
          2);
    }

    debug_image_pub_.publish(
        cv_bridge::CvImage(temp->header, sensor_msgs::image_encodings::BGR8, debug_image).toImageMsg());
  }

}  // namespace rm_radarplugin
