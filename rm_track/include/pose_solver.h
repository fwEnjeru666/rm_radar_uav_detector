/*
 * @Author: fwEnjeru666 enjeru2121@gmail.com
 * @Date: 2026-02-05 16:49:04
 * @LastEditors: fwEnjeru666 enjeru2121@gmail.com
 * @LastEditTime: 2026-04-03 18:34:44
 * @FilePath: /radar_detection_moduel/src/rm_radarplugin/rm_track/include/pose_solver.h
 * @Description: pose solver.h
 * 
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <thread>
#include <rm_msgs/TrackData.h>
#include <rm_radar_msgs/DroneDetection.h>
#include <rm_radar_msgs/DroneDetectionArray.h>
#include <dynamic_reconfigure/server.h>
#include <rm_track/pose_solverConfig.h>




namespace rm_radarplugin
{
    inline double huberLoss(double error, double delta = 3.0) 
    {
        double abs_error = std::abs(error);
        if (abs_error <= delta) 
            return 0.5 * error * error;
        else 
            return delta * (abs_error - 0.5 * delta);
    }
    struct YawOptimizeParams
    {
        bool enable_reprojection{true};  
        bool optimize_translation{true}; 
        double fixed_pitch{-15.0 * CV_PI / 180.0};
        double prior_weight{3.0};
        double huber_delta_px{5.0};
        double huber_delta_deg{30.0};
        double big_range{60.0};
        double big_step{2.0};
        double small_range{3.0};
        double small_step{0.2};
    };

    class PoseSolver
    {
    public:
        PoseSolver() = default;
        ~PoseSolver() = default;
        
        void onInit(ros::NodeHandle &nh);
        void initialize(ros::NodeHandle &nh);
        bool solvePose(); 
        double calReprojectCost(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, 
                                const cv::Mat& K, const cv::Mat& D, 
                                double yaw, double yaw_center);
        void optimizeTranslation(const Eigen::Matrix3d& R);
        void optimizeYaw();
        void PoseSolverCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection);
        void processSingleDetection(const rm_radar_msgs::DroneDetection& detection);
        void publishTransform(const rm_radar_msgs::DroneDetection& detection);
        Eigen::Quaterniond getQuaternion() const { return optimized_q_; };
        Eigen::Vector3d getTranslation() const { return tvec_; };
        

    private:

        // ros
        std::thread my_thread_;
        ros::NodeHandle nh_;
        
        //target sub
        ros::Subscriber track_data_sub_;
        
        //solved pose pub (发布给Tracker)
        ros::Publisher solved_pose_pub_;

        // cam info
        bool has_cam_info_ = false;
        std::string cam_info_topic_;
        ros::Subscriber cam_info_sub_;
        sensor_msgs::CameraInfoConstPtr camera_info_;
        cv::Mat intrinsics_;
        cv::Mat dist_coeffs_;
        void camInfoCB(const sensor_msgs::CameraInfoConstPtr& cam_info)
        {
            if(has_cam_info_) return;
            camera_info_ = cam_info;
            cv::Mat K(3, 3, CV_64F);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    K.at<double>(r, c) = camera_info_->K[r * 3 + c];
            cv::Mat D((int)camera_info_->D.size(), 1, CV_64F);
            for (size_t i = 0; i < camera_info_->D.size(); ++i)
                D.at<double>((int)i, 0) = camera_info_->D[i];
            intrinsics_ = K.clone();
            dist_coeffs_ = D.clone();
            has_cam_info_ = true;
            ROS_INFO("Camera info received and processed.");
        }

        //point 3d
        //单位是米(m)，50mm = 0.050m，12mm = 0.012m
        double armor_half_w_ = 0.012 / 2.0;   // 宽 12mm (水平方向)
        double armor_half_h_ = 0.050 / 2.0;   // 高 50mm (竖直方向)

        std::vector<cv::Point3d> armor_3d_points_{
            cv::Point3d(-armor_half_w_, -armor_half_h_, 0),  // TL (左上)
            cv::Point3d( armor_half_w_, -armor_half_h_, 0),  // TR (右上)
            cv::Point3d( armor_half_w_,  armor_half_h_, 0),  // BR (右下)
            cv::Point3d(-armor_half_w_,  armor_half_h_, 0)   // BL (左下)
        };
        //point 2d
        std::vector<cv::Point2d> armor_2d_points_;

        //solve pnp
        Eigen::Quaterniond q_ = Eigen::Quaterniond::Identity();
        Eigen::Vector3d tvec_ = Eigen::Vector3d::Zero();
        cv::Mat rvec_;

        inline double getAngle(double x1, double y1, double x2, double y2) 
        {
            double dot = x1 * x2 + y1 * y2;
            double det = x1 * y2 - y1 * x2;
            return std::abs(std::atan2(det, dot));
        }

        //get reproject params
        YawOptimizeParams opt_params_;
        // make these writable members (initialized in getReprojectParams)
        double fx_{0}, fy_{0}, cx_{0}, cy_{0}, k1_{0}, k2_{0}, p1_{0}, p2_{0}, k3_{0};
        std::vector<cv::Point2d> undistorted_points_;
        void getReprojectParams()
        {
            if (intrinsics_.empty() || dist_coeffs_.empty()) {
                ROS_WARN("intrinsics or dist_coeffs empty in getReprojectParams");
                return;
            }
            if (armor_2d_points_.size() != 4) {
                ROS_WARN("armor_2d_points_ size is %lu, expected 4", armor_2d_points_.size());
                return;
            }
            fx_ = intrinsics_.at<double>(0, 0);
            fy_ = intrinsics_.at<double>(1, 1);
            cx_ = intrinsics_.at<double>(0, 2);
            cy_ = intrinsics_.at<double>(1, 2);
            k1_ = dist_coeffs_.at<double>(0, 0);
            k2_ = dist_coeffs_.rows > 1 ? dist_coeffs_.at<double>(1, 0) : 0.0;
            p1_ = dist_coeffs_.rows > 2 ? dist_coeffs_.at<double>(2, 0) : 0.0;
            p2_ = dist_coeffs_.rows > 3 ? dist_coeffs_.at<double>(3, 0) : 0.0;
            k3_ = dist_coeffs_.rows > 4 ? dist_coeffs_.at<double>(4, 0) : 0.0;

            // undistort points to normalized coordinates (no P matrix = normalized)
            undistorted_points_.clear();
            cv::undistortPoints(armor_2d_points_, undistorted_points_, intrinsics_, dist_coeffs_);
        }

       //optimize translation/ Quaternion
    //    Eigen::Vector3d optimized_tvec_;
       Eigen::Quaterniond optimized_q_ = Eigen::Quaterniond::Identity();
       
       // ---- PoseSolver dynamic reconfigure (EMA on/off + params) ----
       std::unique_ptr<dynamic_reconfigure::Server<rm_track::pose_solverConfig>> pose_solver_cfg_srv_{nullptr};
       dynamic_reconfigure::Server<rm_track::pose_solverConfig>::CallbackType pose_solver_cfg_cb_;

       bool use_ema_{true};

       void poseSolverConfigCB(rm_track::pose_solverConfig& config, uint32_t level)
       {
           use_ema_ = config.use_ema;
           verbose_log_ = config.verbose_log;
           ema_alpha_ = config.ema_alpha;
           spike_threshold_ = config.spike_threshold;
           max_consecutive_spikes_ = config.max_consecutive_spikes;

           // turning EMA back on should start from current measurement
           if (use_ema_) {
               resetEMA();
           }

           ROS_INFO("[PoseSolver] Reconfigure: use_ema=%d, verbose_log=%d, ema_alpha=%.3f, spike_threshold=%.3f, max_consecutive_spikes=%d",
                    (int)use_ema_, (int)verbose_log_, ema_alpha_, spike_threshold_, max_consecutive_spikes_);
       }
       
       bool verbose_log_ = true;  // 默认启用详细日志
       // ---- end dynamic reconfigure ----

       // ---- EMA smoothing & spike rejection ----
       bool ema_initialized_ = false;
       Eigen::Vector3d ema_tvec_ = Eigen::Vector3d::Zero();
       Eigen::Quaterniond ema_q_ = Eigen::Quaterniond::Identity();
       double ema_alpha_ = 0.3;          // 0→全用历史, 1→不平滑. 0.3 是个好起点
       double spike_threshold_ = 0.5;    // 帧间位移 > 此值(m) 视为毛刺, 用上一帧
       int spike_count_ = 0;
       int max_consecutive_spikes_ = 5;  // 连续毛刺超过此数 → 认为目标真的移动了, 重置EMA

       Eigen::Vector3d applyEMA(const Eigen::Vector3d& raw_t, const Eigen::Quaterniond& raw_q)
       {
           // Bypass if disabled
           if (!use_ema_) {
               ema_q_ = raw_q;
               ema_tvec_ = raw_t;
               ema_initialized_ = false; // avoid using stale history when re-enabled
               spike_count_ = 0;
               return raw_t;
           }

           if (!ema_initialized_)
           {
               ema_tvec_ = raw_t;
               ema_q_ = raw_q;
               ema_initialized_ = true;
               spike_count_ = 0;
               return raw_t;
           }

           double jump = (raw_t - ema_tvec_).norm();
           if (jump > spike_threshold_)
           {
               spike_count_++;
               if (spike_count_ >= max_consecutive_spikes_)
               {
                   // 连续多帧都"跳"到新位置 → 目标确实移动了, 重置
                   ROS_WARN("[PoseSolver] %d consecutive jumps (%.3fm), resetting EMA to new position",
                            spike_count_, jump);
                   ema_tvec_ = raw_t;
                   ema_q_ = raw_q;
                   spike_count_ = 0;
                   return ema_tvec_;
               }
               else
               {
                   ROS_WARN_THROTTLE(1, "[PoseSolver] Spike rejected: jump=%.3fm > %.3fm (count=%d/%d)",
                                     jump, spike_threshold_, spike_count_, max_consecutive_spikes_);
                   return ema_tvec_;  // 沿用上一帧
               }
           }

           spike_count_ = 0;
           // EMA: smoothed = alpha * new + (1-alpha) * old
           ema_tvec_ = ema_alpha_ * raw_t + (1.0 - ema_alpha_) * ema_tvec_;
           // Quaternion SLERP for rotation smoothing
           ema_q_ = ema_q_.slerp(ema_alpha_, raw_q);
           ema_q_.normalize();
           return ema_tvec_;
       }

       void resetEMA()
       {
           ema_initialized_ = false;
           spike_count_ = 0;
       }
       // ---- end EMA ----
       
       //TF broadcaster (moved inside class)
       std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    };


} // namespace rm_radarplugin