#pragma once

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <rm_radar_msgs/DroneDetection.h>
#include <rm_radar_msgs/DroneTrackData.h>
#include <rm_msgs/TrackData.h>
#include <thread>
#include <mutex>
#include <nodelet/nodelet.h>
#include <pluginlib/class_loader.h>
#include <pluginlib/class_list_macros.h>
#include <memory>
#include <ros/callback_queue.h>
#include <dynamic_reconfigure/server.h>
#include <rm_track/trackConfig.h>
#include <pose_solver.h>

#include "ukf.h"
#include "aimm.h"
#include "common.h"

namespace rm_radarplugin
{
    class Tracker : public nodelet::Nodelet
    {
    public:
        Tracker() = default;
        ~Tracker() 
        {
            if (my_thread_.joinable())
                my_thread_.join();
        };
        void onInit();
        void initialize(ros::NodeHandle &nh);
        void camDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection);
        void lidarDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection);  // LiDAR 检测回调

    private:

        ros::Subscriber detection_sub_; // 订阅视觉
        ros::Subscriber lidar_detection_sub_;  // 订阅 LiDAR 检测
        ros::Publisher tracker_pub_;
        ros::CallbackQueue my_queue_;

        bool debug_mode_{false};

        //
        std::thread my_thread_;
        ros::NodeHandle nh_;

        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
        std::string target_frame_;     

        //time
        ros::Time last_time_;
        ros::Time current_time_;

        //ukf
        UKF ukf_;
        void init_ukf(const Eigen::Vector3d& z_meas);

        // AIMM (Adaptive Interacting Multiple Model)
        AIMM aimm_;
        bool use_aimm_{false};
        void init_aimm(const Eigen::Vector3d& z_meas);

        // Unified filter interface helpers
        bool filterInitialized() const;
        void filterSetDt(double dt);
        void filterPredict();
        void filterUpdate(const Eigen::Vector3d& z_meas);
        Eigen::VectorXd filterGetState() const;
        Eigen::MatrixXd filterGetBaseMeasurementNoise() const;
        void filterSetMeasurementNoise(const Eigen::MatrixXd& R);
        bool filterIsInitialized() const;

        //pub
        rm_msgs::TrackData track_data_msg_;

        //transform

        bool transform2Odom(const std_msgs::Header& header, const Eigen::Vector3d& p_cam, Eigen::Vector3d& p_odom);
        

        //state
        int detected_count_{};
        int detect_fail_count_{};  // DETECTING 状态下连续无数据计数
        int lost_count_{};
        int hit_threshold_{};
        int max_lost_count_{};

        // tracking state - 初始状态 LOST
        rm_track::TrackState track_state_{rm_track::LOST};
        bool is_tracking_{false};
        uint8_t track_id_{0};

        // time + publisher already declared above

        //state handling
        void handleDetectingState(bool has_data, const Eigen::Vector3d& z_meas, double dt);
        void handleTrackingState(bool has_data, const Eigen::Vector3d& z_meas, double dt);
        void handleTempLostState(bool has_data, const Eigen::Vector3d& z_meas, double dt);
        void handleLostState(bool has_data, const Eigen::Vector3d& z_meas, double dt);

        void publishTrackerData();
        void processTracking(bool has_data, const Eigen::Vector3d& z_meas, double dt);
        // init_ukf already declared above

        //dynamic reconfigure
        dynamic_reconfigure::Server<rm_track::trackConfig>* track_cfg_srv_;
        dynamic_reconfigure::Server<rm_track::trackConfig>::CallbackType track_cfg_cb_;
        void trackconfigCB(rm_track::trackConfig& config, uint32_t level);

        //Quaternion - 初始化为单位四元数，防止发布无效TF
        tf2::Quaternion q_track_{0, 0, 0, 1};  // identity quaternion
        bool q_track_valid_{false};  // 标记四元数是否有效
        std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

        //pose solver
        PoseSolver pose_solver_;
        
        // LiDAR 检测融合相关
        rm_radar_msgs::DroneDetection::ConstPtr latest_lidar_detection_;
        ros::Time last_lidar_time_;
        ros::Time last_vision_time_;
        double lidar_timeout_{0.5};  // LiDAR 检测超时时间 (秒)
        double vision_timeout_{0.3}; // 视觉检测超时时间 (秒)
        bool use_lidar_fusion_{true};  // 是否启用 LiDAR 融合
        std::mutex lidar_mutex_;

        //cam 
        std::mutex cam_mutex_;
        rm_radar_msgs::DroneDetection::ConstPtr latest_cam_detection_;
        ros::Time last_cam_time_;
        

        //
        ros::Timer detection_timer_;  // 检测定时器，用于触发融合
        void detectionCB(const ros::TimerEvent& event);



    };

}