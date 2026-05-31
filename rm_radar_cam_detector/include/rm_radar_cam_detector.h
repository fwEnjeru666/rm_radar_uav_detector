#pragma once

#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <rm_radar_msgs/DroneTrackData.h>
#include <rm_radar_cam_detector/CamDetectorConfig.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <tf2_ros/transform_listener.h>

namespace rm_radarplugin
{
  class Bbox2D
  {
    public:
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

      Bbox2D() = default;
      explicit Bbox2D(const Bbox& bbox) : bbox2d(bbox) {}
      void clear()
      {
        bbox2d.is_valid = false;
        bbox2d.need_roi = false;
        bbox2d.bbox_min_pt = cv::Point2f(-1.0f, -1.0f);
        bbox2d.bbox_max_pt = cv::Point2f(-1.0f, -1.0f);
        bbox2d.target_pt = cv::Point2f(-1.0f, -1.0f);
        bbox2d.bbox_area = cv::Rect();
        bbox2d.roi_area = cv::Rect();
        bbox2d.roi_x = -1;
        bbox2d.roi_y = -1;
        bbox2d.expand_ratio = 0.0f;
      }

  };



  class CamDetector : public nodelet::Nodelet
  {
    public:
      CamDetector() = default;
      ~CamDetector() override = default;

      void onInit() override;

    private:
    //track callback
      std::mutex track_mutex_;
      std::deque<rm_radar_msgs::DroneTrackData> track_buffer_;
      static constexpr std::size_t kTrackBufferMaxSize = 200;
      void trackCallback(const rm_radar_msgs::DroneTrackData::ConstPtr& msg)
      {
        std::lock_guard<std::mutex> lock(track_mutex_);
        track_buffer_.push_back(*msg);
        while (track_buffer_.size() > kTrackBufferMaxSize)
        {
          track_buffer_.pop_front();
        }
      }

    //image callback
      image_transport::CameraSubscriber cam_sub_;
      image_transport::Publisher debug_image_pub_;

      cv::Mat intrinsics_;
      cv::Mat dist_coeffs_;
      sensor_msgs::CameraInfoConstPtr camera_info_;
      std::mutex camera_model_mutex_;
      double fx_ ;
      double fy_ ;
      double cx_ ; // 真实的图像中心 x
      double cy_ ; // 真实的图像中心 
      cv::Point2d image_center_; // 计算得到的图像中心
      bool camera_model_initialized_{false};
      int image_width_{-1};
      int image_height_{-1};
      void cam_callback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info)
      {
          {
            std::lock_guard<std::mutex> lock(camera_model_mutex_);
            camera_info_ = info;
            if (!camera_model_initialized_)
            {
                intrinsics_ = cv::Mat(3, 3, CV_64F, (void*)info->K.data()).clone();
                dist_coeffs_ = cv::Mat(info->D).clone();
                camera_model_initialized_ = true;
                fx_ = intrinsics_.at<double>(0, 0);
                fy_ = intrinsics_.at<double>(1, 1);
                cx_ = intrinsics_.at<double>(0, 2);
                cy_ = intrinsics_.at<double>(1, 2);
                image_center_ = cv::Point2d(cx_, cy_);
                image_width_ = info->width;
                image_height_ = info->height;

                ROS_INFO(
                    "[rm_radar_cam_detector]Camera intrinsics initialized!\n"
                    "  fx=%.2f, fy=%.2f\n"
                    "  cx=%.2f, cy=%.2f\n"
                    "  image_width=%d, image_height=%d",
                    fx_, fy_, cx_, cy_, image_width_, image_height_);
            }
          }
          if (image_width_ <= 0 || image_height_ <= 0)
          {
            return;
          }

          cv_bridge::CvImageConstPtr temp = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
          if (!temp || temp->image.empty())
          {
            return;
          }

          bool has_track = false;
          rm_radar_msgs::DroneTrackData track_snapshot;
          double timeout_s = 0.2;
          double tf_timeout_s = 0.02;
          bool debug_mode = false;
          bool need_roi = false;
          float expand_ratio = 0.0f;
          {
            std::lock_guard<std::mutex> lock(track_mutex_);
            if (!track_buffer_.empty())
            {
              has_track = true;
              const ros::Time image_stamp = img->header.stamp;
              double best_abs_dt = std::numeric_limits<double>::infinity();
              for (const auto& track : track_buffer_)
              {
                if (track.header.stamp.isZero())
                {
                  continue;
                }
                const double dt = (image_stamp - track.header.stamp).toSec();
                const double abs_dt = std::abs(dt);
                if (abs_dt < best_abs_dt)
                {
                  best_abs_dt = abs_dt;
                  track_snapshot = track;
                }
              }
              if (track_snapshot.header.stamp.isZero())
              {
                has_track = false;
              }
            }
          }
          {
            std::lock_guard<std::mutex> lock(param_mutex_);
            timeout_s = timeout_s_;
            tf_timeout_s = tf_timeout_s_;
            debug_mode = debug_mode_;
            need_roi = need_roi_;
            expand_ratio = expand_ratio_;
          }
          if (debug_mode)
          {
            ROS_INFO_THROTTLE(
                1.0,
                "[rm_radar_cam_detector][1] cam_callback start: image_stamp=%.3f has_track=%s",
                img->header.stamp.toSec(),
                has_track ? "true" : "false");
          }

          std::string status_text;
          if (!has_track)
          {
            if (debug_mode)
            {
              ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][2] No track data received yet.");
            }
            status_text = "NO TRACK";
            drawBbox(temp, nullptr, status_text);
            return;
          }
          if (debug_mode)
          {
            ROS_INFO_THROTTLE(1.0, "[rm_radar_cam_detector][2] Track data exists.");
          }
          if (img->header.frame_id.empty())
          {
            if (debug_mode)
            {
              ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][3] Image frame_id is empty.");
            }
            status_text = "IMAGE FRAME_ID EMPTY";
            drawBbox(temp, nullptr, status_text);
            return;
          }
          {
            const double track_age_s = (img->header.stamp - track_snapshot.header.stamp).toSec();
            if (std::abs(track_age_s) > timeout_s)
            {
              if (debug_mode)
              {
                if (std::abs(track_age_s) > 10.0)
                {
                  ROS_WARN_THROTTLE(
                      1.0,
                      "[rm_radar_cam_detector][3] Large time-domain mismatch (dt=%.3fs). Check /use_sim_time and /clock across camera+lidar.",
                      track_age_s);
                }
                ROS_WARN_THROTTLE(
                    1.0,
                    "[rm_radar_cam_detector][3] Unsynced track/image (dt=%.3fs, |dt|>%.3fs), skip frame.",
                    track_age_s,
                    timeout_s);
              }
              status_text = "STALE TRACK";
              drawBbox(temp, nullptr, status_text);
              return;
            }
          }
          if (debug_mode)
          {
            ROS_INFO_THROTTLE(1.0, "[rm_radar_cam_detector][3] Track age check passed.");
          }
          if (!projectBboxToImage(track_snapshot,
                                  img->header.stamp,
                                  img->header.frame_id,
                                  tf_timeout_s,
                                  need_roi,
                                  expand_ratio,
                                  debug_mode,
                                  intrinsics_,
                                  dist_coeffs_,
                                  image_width_,
                                  image_height_))
          {
            status_text = "PROJECTION FAILED";
            drawBbox(temp, nullptr, status_text);
            return;
          }
          if (debug_mode)
          {
            ROS_INFO_THROTTLE(1.0, "[rm_radar_cam_detector][4] Bbox projection succeeded.");
          }

          Bbox2D::Bbox bbox;
          {
            std::lock_guard<std::mutex> lock(bbox_mutex_);
            bbox = drone_bbox_.bbox2d;
          }
          if (bbox.need_roi && bbox.roi_area.area() > 0)
          {
            const cv::Rect image_rect(0, 0, temp->image.cols, temp->image.rows);
            const cv::Rect roi_rect = bbox.roi_area & image_rect;
            if (roi_rect.area() > 0)
            {
              cv::Mat roi_image = temp->image(roi_rect).clone();
              roi_image_pub_.publish(cv_bridge::CvImage(img->header, sensor_msgs::image_encodings::BGR8, roi_image).toImageMsg());
              if (debug_mode)
              {
                ROS_INFO_THROTTLE(
                    1.0,
                    "[rm_radar_cam_detector][5] ROI published: x=%d y=%d w=%d h=%d",
                    roi_rect.x,
                    roi_rect.y,
                    roi_rect.width,
                    roi_rect.height);
              }
            }
            else if (debug_mode)
            {
              ROS_WARN_THROTTLE(1.0, "[rm_radar_cam_detector][5] ROI rect became invalid after clipping.");
            }
          }
          else if (debug_mode)
          {
            ROS_INFO_THROTTLE(1.0, "[rm_radar_cam_detector][5] ROI not requested or empty, skip ROI publish.");
          }
          drawBbox(temp, &bbox, status_text);
          if (debug_mode)
          {
            ROS_INFO_THROTTLE(1.0, "[rm_radar_cam_detector][6] Debug image draw/publish finished.");
          }
      }

      bool projectBboxToImage(const rm_radar_msgs::DroneTrackData& track,
                              const ros::Time& image_stamp,
                              const std::string& target_frame,
                              double tf_timeout_s,
                              bool need_roi,
                              float expand_ratio,
                              bool debug_mode,
                              const cv::Mat& intrinsics,
                              const cv::Mat& dist_coeffs,
                              int image_width,
                              int image_height);
      bool isValid(const std::vector<cv::Point2f>& projected_points,
                   const cv::Point2f& projected_target_point,
                   bool debug_mode,
                   bool need_roi,
                   float expand_ratio,
                   int image_width,
                   int image_height);
      void reconfigCB(rm_radar_cam_detector::CamDetectorConfig& config, uint32_t level);
      void drawBbox(const cv_bridge::CvImageConstPtr& temp, const Bbox2D::Bbox* bbox, const std::string& status_text);
    
      ros::NodeHandle nh_;
      ros::NodeHandle pnh_;

      std::shared_ptr<image_transport::ImageTransport> it_;
      image_transport::CameraSubscriber image_sub_;
      image_transport::Publisher roi_image_pub_;
      ros::Subscriber track_sub_;

      mutable tf2_ros::Buffer tf_buffer_{ros::Duration(10.0)};
      std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
      std::unique_ptr<dynamic_reconfigure::Server<rm_radar_cam_detector::CamDetectorConfig>> cfg_server_;
      dynamic_reconfigure::Server<rm_radar_cam_detector::CamDetectorConfig>::CallbackType cfg_cb_;

      std::mutex param_mutex_;

      std::string image_topic_;
      std::string track_topic_;
      std::string roi_topic_;
      std::string debug_topic_{"debugimage"};
      std::string fixed_frame_{"base_link"};

      double timeout_s_{0.2};
      double tf_timeout_s_{0.02};
      int line_thickness_{2};
      bool debug_mode_{false};
      bool disable_{false};
      bool draw_bbox_{true};
      bool draw_roi_{true};
      double pixel_bias_u_{0.0};
      double pixel_bias_v_{0.0};

      // bbox2d
      std::mutex bbox_mutex_;
      Bbox2D drone_bbox_;
      bool need_roi_{false};
      float expand_ratio_{0.0f};
  };

}  // namespace rm_radarplugin
