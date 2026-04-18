#pragma once

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/CameraInfo.h>
#include <rm_radar_msgs/DroneDetection.h>
#include <rm_radar_msgs/DroneTrackData.h>
#include <rm_msgs/TrackData.h>
#include <rm_msgs/GimbalCmd.h>
#include <thread>
#include <mutex>
#include <array>
#include <nodelet/nodelet.h>
#include <pluginlib/class_loader.h>
#include <pluginlib/class_list_macros.h>
#include <memory>
#include <ros/callback_queue.h>
#include <dynamic_reconfigure/server.h>
#include <rm_track/trackConfig.h>

#include "ukf.h"
#include "aimm.h"
#include "common.h"
#include <visualization_msgs/MarkerArray.h>
#include <rm_radar_msgs/UavDetectionArray.h>
#include <rm_radar_msgs/aimm_debugger.h>

namespace rm_radarplugin
{
    class Tracker : public nodelet::Nodelet
    {
    public:
        Tracker();
        ~Tracker() 
        {
            if (my_thread_.joinable())
                my_thread_.join();
        };
        void onInit();
        void camInfoCB(const sensor_msgs::CameraInfoConstPtr& cam_info);
        void teleDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection);
        void wideDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection);
        void lidarDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection);  // LiDAR 检测回调
        void detectionCB(const ros::TimerEvent& event);  // 定时器回调，触发融合

        // Debug publisher
        ros::Publisher aimm_debug_pub_;

    private:
        rm_msgs::TrackData target_state;
        Eigen::VectorXd x_state_;

        ros::NodeHandle nh_;
        //tele cam
        ros::Subscriber tele_cam_info_sub_;
        ros::Subscriber tele_detection_sub_; // 订阅tele视觉

        //wide cam
        ros::Subscriber wide_cam_info_sub_;
        ros::Subscriber wide_detection_sub_; // 订阅wide视觉


        //lidar
        ros::Subscriber lidar_detection_sub_;  // 订阅 LiDAR 检测

        ros::Publisher tracker_pub_;
        ros::Publisher img_tracker_pub_;

        ros::CallbackQueue my_queue_;

        bool debug_mode_{false};
        bool aimm_debug_mode_{false};
        bool state_log_mode_{true};

        //
        std::thread my_thread_;

        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
        std::string target_frame_;     

        //time
        ros::Time last_time_;
        ros::Time current_time_;

        // UKF for pure UKF mode
        std::shared_ptr<UKF> ukf_;
        int pure_ukf_model_idx_ = 0; // Index for the model used in pure UKF mode
        int pure_ukf_model_selector_ = 0; // Dynamic reconfigure selector value (CV/CA/CTRV/SINGER)
        int resolvePureUkfModelIndex(int selector, bool warn_if_fallback) const;

        // AIMM (Adaptive Interacting Multiple Model)
        std::shared_ptr<AIMM> aimm_;
        bool use_aimm_{false};

        // Model Management
        std::shared_ptr<model_register> model_manager_;
        std::vector<std::shared_ptr<BaseModel>> loaded_models_;
        std::vector<std::string> loaded_model_names_; // Unified filter interface helpers
        bool filterInitialized() const;
        void filterSetDt(double dt);
        void filterPredict();
        void filterUpdate(const Eigen::VectorXd& z_meas);
        Eigen::VectorXd filterGetState() const;
        void filterSetMeasurementNoise(const Eigen::MatrixXd& R);
        bool filterIsInitialized() const;
        bool validatePixelMeasurement(const Eigen::VectorXd& z_meas,
                          double distance,
                          double& nis,
                          double& rmse,
                          double& rmse_gate) const;

        //pub
        // rm_msgs::GimbalCmd track_data_msg_;

        //transform

        bool transform2Odom(const std_msgs::Header& header, const Eigen::Vector3d& p_cam, Eigen::Vector3d& p_odom);
        bool getProjectionExtrinsic(const std_msgs::Header& header);
        void loadProjectionParams();


        struct CameraIntrinsics {
        double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
        std::array<double, 9> K{};
        std::vector<double> D;
        };    
        std::map<std::string, CameraIntrinsics> camera_intrinsics_map_;

        void applyProjectionParamsToModels(const CameraIntrinsics& intrinsics);
        std::string last_active_frame_id_;

        //state
        int detected_count_{};
        int detect_fail_count_{};  // DETECTING 状态下连续无数据计数
        int lost_count_{};
        int hit_threshold_{};
        int max_lost_count_{};
        int cam_hitcount_{};
        int cam_lostcount_{};
        int lidar_hitcount_{};
        int lidar_lostcount_{};
        int cam_detected_count_{};
        int lidar_detected_count_{};
        int cam_lost_counter_{};
        int lidar_lost_counter_{};
        bool tracking_with_cam_{true};

        // tracking state - 初始状态 LOST
        rm_track::TrackState track_state_{rm_track::LOST};
        bool is_tracking_{false};

        // time + publisher already declared above

        //state handling
        void handleDetectingState(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt);
        void handleTrackingState(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt);
        void handleTempLostState(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt);
        void handleLostState(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt);

        void publishTrackerData();
        void processTracking(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt);
        // init_ukf already declared above

        //dynamic reconfigure
        std::unique_ptr<dynamic_reconfigure::Server<rm_track::trackConfig>> track_cfg_srv_;
        dynamic_reconfigure::Server<rm_track::trackConfig>::CallbackType track_cfg_cb_;
        void trackconfigCB(rm_track::trackConfig& config, uint32_t level);

        //Quaternion - 初始化为单位四元数，防止发布无效TF
        tf2::Quaternion q_track_{0, 0, 0, 1};  // identity quaternion
        bool q_track_valid_{false};  // 标记四元数是否有效
        std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

        
        // LiDAR 检测融合相关
        rm_radar_msgs::DroneDetection::ConstPtr latest_lidar_detection_;
        ros::Time last_lidar_time_;
        ros::WallTime last_lidar_receive_time_;
        ros::WallTime last_cam_receive_time_;
        double lidar_timeout_{0.5};  // LiDAR 检测超时时间 (秒)
        double vision_timeout_{0.3}; // 视觉检测超时时间 (秒)

        mutable std::mutex noise_mtx_;  
        void getPixelMeasurementNoise();
        double tele_pixel_noise_u_{4.0};
        double tele_pixel_noise_v_{4.0};
        double wide_pixel_noise_u_{4.0};
        double wide_pixel_noise_v_{4.0};
        Eigen::MatrixXd R_tele_pixel_;
        Eigen::MatrixXd R_wide_pixel_;

        void get3DMeasurementNoise();
        Eigen::Matrix3d R_tele_;
        Eigen::Matrix3d R_wide_;
        Eigen::Matrix3d R_lidar_;
        double tele_r_pos_xy_{};
        double tele_r_pos_z_{};
        double tele_r_yaw_{};
        double wide_r_pos_xy_{};
        double wide_r_pos_z_{};
        double wide_r_yaw_{};
        double lidar_r_pos_xy_{};
        double lidar_r_pos_z_{};
        double lidar_r_yaw_{};
        bool force_3d_measurement_{false};
        bool enable_pixel_gating_{true};
        bool enable_3d_fallback_{true};
        double pixel_nis_gate_{40.0};
        double pixel_reproj_rmse_gate_near_{12.0};
        double pixel_reproj_rmse_gate_mid_{20.0};
        double pixel_reproj_rmse_gate_far_{30.0};
        double pixel_gate_distance_near_{8.0};
        double pixel_gate_distance_far_{18.0};
        bool use_lidar_fusion_{true};  // 是否启用 LiDAR 融合
        std::mutex lidar_mutex_;

        //cam 
        std::mutex tele_cam_mutex_;
        std::mutex wide_cam_mutex_;

        rm_radar_msgs::DroneDetection::ConstPtr latest_tele_cam_detection_;
        rm_radar_msgs::DroneDetection::ConstPtr latest_wide_cam_detection_;
        
        ros::WallTime last_tele_receive_time_;
        ros::WallTime last_wide_receive_time_;
        
        bool processVisionMeasurement(const rm_radar_msgs::DroneDetection::ConstPtr& cam_msg, 
                                  Eigen::VectorXd& z_meas, Eigen::Vector3d& z_init, std::string& active_frame_id);

        //
        ros::Timer detection_timer_;  // 检测定时器，用于触发融合

        struct ProjectionParams
        {
            double fx{1.0};
            double fy{1.0};
            double cx{0.0};
            double cy{0.0};
            double armor_width{0.135};
            double armor_height{0.055};
        } projection_params_;
        std::mutex projection_mutex_;
        std::string cam_info_topic_{"/hk_camera/camera_info"};
        bool has_camera_info_{false};
        std::array<double, 9> camera_k_{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        std::vector<double> camera_d_;

        // Last known valid position and velocity for LOST state
        Eigen::Vector3d last_known_position_{0, 0, 0};
        Eigen::Vector3d last_known_velocity_{0, 0, 0};


    };

}