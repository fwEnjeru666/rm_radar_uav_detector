/*
 * @Author: fwEnjeru666 enjeru2121@gmail.com
 * @Date: 2026-02-05 16:49:04
 * @LastEditors: fwEnjeru666 enjeru2121@gmail.com
 * @LastEditTime: 2026-02-07 21:11:41
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
        bool optimize_translation{false};  // Disabled - was causing bad results
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
        void PoseSolverCB(const rm_radar_msgs::DroneDetectionArray::ConstPtr& detections);
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
        double sz = 0.133 / 2.0; // armor size 130mm x 130mm
        std::vector<cv::Point3d> armor_3d_points_{
            cv::Point3d(-sz, -sz, 0),
            cv::Point3d(-sz, sz, 0),
            cv::Point3d(sz, sz, 0),
            cv::Point3d(sz, -sz, 0)
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
       
       //TF broadcaster (moved inside class)
       std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    };


} // namespace rm_radarplugin