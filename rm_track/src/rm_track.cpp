#include <rm_track.h>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cctype>
#include <limits>

PLUGINLIB_EXPORT_CLASS(rm_radarplugin::Tracker, nodelet::Nodelet);

namespace rm_radarplugin
{
    namespace
    {
        std::string toUpper(std::string s)
        {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            return s;
        }
    }

    Tracker::Tracker(){}

    int Tracker::resolvePureUkfModelIndex(int selector, bool warn_if_fallback) const
    {
        if (loaded_models_.empty())
        {
            return 0;
        }

        const std::vector<std::string> selector_names = {"CV", "CA", "CTRV", "SINGER"};
        if (selector >= 0 && selector < static_cast<int>(selector_names.size()))
        {
            const std::string wanted = selector_names[selector];
            for (size_t i = 0; i < loaded_model_names_.size(); ++i)
            {
                if (toUpper(loaded_model_names_[i]) == wanted)
                {
                    return static_cast<int>(i);
                }
            }
        }

        // Backward-compatible fallback: treat selector as direct index.
        if (selector >= 0 && selector < static_cast<int>(loaded_models_.size()))
        {
            return selector;
        }

        if (warn_if_fallback)
        {
            ROS_WARN("[Tracker] pure_ukf_model_type=%d cannot be resolved from loaded models. Falling back to model index 0 (%s).",
                     selector,
                     loaded_model_names_.empty() ? "unknown" : loaded_model_names_[0].c_str());
        }
        return 0;
    }

    void Tracker::loadProjectionParams()
    {
        std::lock_guard<std::mutex> lock(projection_mutex_);
        // Intrinsics are sourced from CameraInfo at runtime.
        projection_params_.fx = 0.0;
        projection_params_.fy = 0.0;
        projection_params_.cx = 0.0;
        projection_params_.cy = 0.0;
        projection_params_.armor_width = nh_.param("armor_width", projection_params_.armor_width);
        projection_params_.armor_height = nh_.param("armor_height", projection_params_.armor_height);
    }

    void Tracker::applyProjectionParamsToModels()
    {
        std::lock_guard<std::mutex> lock(projection_mutex_);
        for (auto& model : loaded_models_)
        {
            if (!model)
            {
                continue;
            }
            model->setProjectionParams(
                projection_params_.fx,
                projection_params_.fy,
                projection_params_.cx,
                projection_params_.cy,
                projection_params_.armor_width,
                projection_params_.armor_height);
        }
    }

    void Tracker::camInfoCB(const sensor_msgs::CameraInfoConstPtr& cam_info)
    {
        if (!cam_info)
        {
            return;
        }

        const double fx = cam_info->K[0];
        const double fy = cam_info->K[4];
        const double cx = cam_info->K[2];
        const double cy = cam_info->K[5];
        if (fx <= 1e-6 || fy <= 1e-6)
        {
            ROS_WARN_THROTTLE(1.0, "[Tracker] Ignoring invalid CameraInfo intrinsics fx=%.3f fy=%.3f", fx, fy);
            return;
        }

        bool changed = false;
        {
            std::lock_guard<std::mutex> lock(projection_mutex_);
            changed = (!has_camera_info_) ||
                      (std::abs(projection_params_.fx - fx) > 1e-6) ||
                      (std::abs(projection_params_.fy - fy) > 1e-6) ||
                      (std::abs(projection_params_.cx - cx) > 1e-6) ||
                      (std::abs(projection_params_.cy - cy) > 1e-6);
            if (!changed)
            {
                return;
            }

            projection_params_.fx = fx;
            projection_params_.fy = fy;
            projection_params_.cx = cx;
            projection_params_.cy = cy;
            for (size_t i = 0; i < camera_k_.size(); ++i)
            {
                camera_k_[i] = cam_info->K[i];
            }
            camera_d_ = cam_info->D;
            has_camera_info_ = true;
        }

        applyProjectionParamsToModels();
        ROS_INFO_THROTTLE(2.0,
                          "[Tracker] Updated projection intrinsics from CameraInfo: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
                          fx, fy, cx, cy);
    }

    void Tracker::onInit()
    {
        ros::NodeHandle nh = getMTPrivateNodeHandle();
        nh_ = nh;
        ROS_INFO("[Tracker] Initializing nodelet...");
        ROS_INFO("[Tracker] Nodelet name: %s, private namespace: %s",
                 getName().c_str(), nh_.getNamespace().c_str());

        // Parameters from parameter server
        // These are the initial values. They can be changed dynamically.
        use_aimm_ = nh_.param("use_aimm", false);
        target_frame_ = nh_.param("target_frame", std::string("odom"));
        hit_threshold_ = nh_.param("hit_threshold", 3);
        max_lost_count_ = nh_.param("max_lost_count", 5);
        cam_hitcount_ = nh_.param("cam_hitcount", hit_threshold_);
        cam_lostcount_ = nh_.param("cam_lostcount", max_lost_count_);
        lidar_hitcount_ = nh_.param("lidar_hitcount", hit_threshold_);
        lidar_lostcount_ = nh_.param("lidar_lostcount", max_lost_count_);
        debug_mode_ = nh_.param("debug_mode", false);
        aimm_debug_mode_ = nh_.param("aimm_debug_mode", false);
        state_log_mode_ = nh_.param("state_log_mode", true);
        use_lidar_fusion_ = nh_.param("use_lidar_fusion", true);
        lidar_timeout_ = nh_.param("lidar_timeout", 0.5);
        vision_timeout_ = nh_.param("vision_timeout", 0.3);
        pixel_noise_u_ = nh_.param("pixel_noise_u", 4.0);
        pixel_noise_v_ = nh_.param("pixel_noise_v", 4.0);
        force_3d_measurement_ = nh_.param("force_3d_measurement", false);
        enable_pixel_gating_ = nh_.param("enable_pixel_gating", true);
        enable_3d_fallback_ = nh_.param("enable_3d_fallback", true);
        pixel_nis_gate_ = nh_.param("pixel_nis_gate", 40.0);
        pixel_reproj_rmse_gate_near_ = nh_.param("pixel_reproj_rmse_gate_near", 12.0);
        pixel_reproj_rmse_gate_mid_ = nh_.param("pixel_reproj_rmse_gate_mid", 20.0);
        pixel_reproj_rmse_gate_far_ = nh_.param("pixel_reproj_rmse_gate_far", 30.0);
        pixel_gate_distance_near_ = nh_.param("pixel_gate_distance_near", 8.0);
        pixel_gate_distance_far_ = nh_.param("pixel_gate_distance_far", 18.0);
        cam_info_topic_ = nh_.param<std::string>("cam_info_topic", "/hk_camera/camera_info");

        // ROS Interfaces
        cam_info_sub_ = nh_.subscribe(cam_info_topic_, 1, &Tracker::camInfoCB, this);
        detection_sub_ = nh_.subscribe("/pose_solver/solved_pose", 10, &Tracker::camDetectionCB, this);
        lidar_detection_sub_ = nh_.subscribe("/lidar_detector/lidar_detection", 1, &Tracker::lidarDetectionCB, this);
        tracker_pub_ = nh_.advertise<rm_msgs::TrackData>("/tracker/track_data", 1);
        aimm_debug_pub_ = nh_.advertise<rm_radar_msgs::aimm_debugger>("/tracker/aimm_debug", 1);

        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(ros::Duration(10));
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ =  std::make_shared<tf2_ros::TransformBroadcaster>();

        // Timer for processing detections
        detection_timer_ = nh_.createTimer(ros::Duration(0.01), &Tracker::detectionCB, this);  

        // Initialize the model manager and load models
        model_manager_ = std::make_shared<model_register>(nh_);
        if (model_manager_->load_models() == 0)
        {
            ROS_FATAL("[Tracker] No models loaded. Shutting down.");
            ros::shutdown();
            return;
        }
        loaded_models_ = model_manager_->get_all_models();
        loaded_model_names_ = model_manager_->get_model_names();
        loadProjectionParams();
        applyProjectionParamsToModels();
        ROS_INFO("[Tracker] Projection params initialized (armor size only); waiting CameraInfo on %s for intrinsics.",
             cam_info_topic_.c_str());

        // Create filter instances
        aimm_ = std::make_shared<AIMM>(nh_, loaded_models_);
        ROS_INFO("[Tracker] Initialized AIMM with %lu models.", loaded_models_.size());

        // Use initial parameter for pure_ukf_model_type to select the model
        pure_ukf_model_selector_ = nh_.param("pure_ukf_model_type", 0);
        if (loaded_models_.empty()) {
            ROS_ERROR("[Tracker] No models loaded. Cannot initialize UKF.");
            ros::shutdown();
            return;
        }
        pure_ukf_model_idx_ = resolvePureUkfModelIndex(pure_ukf_model_selector_, true);

        ros::NodeHandle pure_ukf_nh(nh_, "pure_ukf");
        ukf_ = std::make_shared<UKF>(pure_ukf_nh, loaded_models_[pure_ukf_model_idx_]);

        // Initialize other components
        pose_solver_.onInit(nh);

        // Initialize dynamic reconfigure server LAST
        track_cfg_srv_.reset(new dynamic_reconfigure::Server<rm_track::trackConfig>(nh_));
        track_cfg_cb_ = boost::bind(&Tracker::trackconfigCB, this, _1, _2);
        track_cfg_srv_->setCallback(track_cfg_cb_);
        ROS_INFO("[Tracker] Dynamic reconfigure attached at namespace: %s", nh_.getNamespace().c_str());

        // Initial status log will be printed by the first trackconfigCB call
        ROS_INFO("[Tracker] Initialized successfully.");
    }

    bool Tracker::validatePixelMeasurement(const Eigen::VectorXd& z_meas,
                                           double distance,
                                           double& nis,
                                           double& rmse,
                                           double& rmse_gate) const
    {
        nis = 1000.0;
        rmse = 1e6;
        rmse_gate = pixel_reproj_rmse_gate_mid_;

        if (z_meas.size() != 8 || !filterIsInitialized())
        {
            return true;
        }

        Eigen::VectorXd z_pred;
        Eigen::MatrixXd S;
        if (use_aimm_)
        {
            if (!aimm_) return true;
            nis = aimm_->computeInnovation(z_meas, z_pred, S);
        }
        else
        {
            if (!ukf_) return true;
            nis = ukf_->computeInnovation(z_meas, z_pred, S);
        }

        if (z_pred.size() != z_meas.size())
        {
            return false;
        }

        if (distance <= pixel_gate_distance_near_)
        {
            rmse_gate = pixel_reproj_rmse_gate_near_;
        }
        else if (distance >= pixel_gate_distance_far_)
        {
            rmse_gate = pixel_reproj_rmse_gate_far_;
        }

        rmse = std::sqrt((z_meas - z_pred).squaredNorm() / static_cast<double>(z_meas.size()));
        return (nis <= pixel_nis_gate_) && (rmse <= rmse_gate);
    }

    void Tracker::trackconfigCB(rm_track::trackConfig& config, uint32_t level)
    {
        // Update basic parameters
        (void)level;
        hit_threshold_ = config.hit_threshold;
        max_lost_count_ = config.max_lost_count;
        cam_hitcount_ = config.cam_hitcount;
        cam_lostcount_ = config.cam_lostcount;
        lidar_hitcount_ = config.lidar_hitcount;
        lidar_lostcount_ = config.lidar_lostcount;
        debug_mode_ = config.debug_mode;
        aimm_debug_mode_ = config.aimm_debug_mode;
        state_log_mode_ = config.state_log_mode;
        pixel_noise_u_ = config.pixel_noise_u;
        pixel_noise_v_ = config.pixel_noise_v;
        force_3d_measurement_ = config.force_3d_measurement;
        enable_pixel_gating_ = config.enable_pixel_gating;
        enable_3d_fallback_ = config.enable_3d_fallback;
        pixel_nis_gate_ = config.pixel_nis_gate;
        pixel_reproj_rmse_gate_near_ = config.pixel_reproj_rmse_gate_near;
        pixel_reproj_rmse_gate_mid_ = config.pixel_reproj_rmse_gate_mid;
        pixel_reproj_rmse_gate_far_ = config.pixel_reproj_rmse_gate_far;
        pixel_gate_distance_near_ = config.pixel_gate_distance_near;
        pixel_gate_distance_far_ = config.pixel_gate_distance_far;

        bool needs_reset = false;
        std::stringstream log_msg;
        log_msg << "[Tracker] Configuration updated. ";

        // Check for estimator type change (AIMM vs UKF)
        if (config.use_aimm != use_aimm_) {
            use_aimm_ = config.use_aimm;
            needs_reset = true;
        }

        // Check for pure UKF model change
        if (config.pure_ukf_model_type != pure_ukf_model_selector_) {
            pure_ukf_model_selector_ = config.pure_ukf_model_type;
            const int resolved_idx = resolvePureUkfModelIndex(pure_ukf_model_selector_, true);
            if (resolved_idx != pure_ukf_model_idx_) {
                pure_ukf_model_idx_ = resolved_idx;
                // If we are currently in pure UKF mode, apply the model change immediately
                if (!use_aimm_) {
                    ukf_->setModel(loaded_models_[pure_ukf_model_idx_]);
                    needs_reset = true;
                }
            }
        }

        if (needs_reset) {
            track_state_ = rm_track::LOST;
            is_tracking_ = false;
            q_track_valid_ = false;
        }

        // Log the current state
        if (use_aimm_) {
            log_msg << "Estimator: AIMM.";
        } else {
            log_msg << "Estimator: UKF, Model: " << loaded_model_names_[pure_ukf_model_idx_] << ".";
        }
        if (needs_reset) {
            log_msg << " Filter state has been reset.";
        }
        log_msg << " Force3D=" << (force_3d_measurement_ ? "ON" : "OFF");
        log_msg << " 3DFallback=" << (enable_3d_fallback_ ? "ON" : "OFF");
        ROS_INFO_STREAM(log_msg.str());
    }

    void Tracker::lidarDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection)
    {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        latest_lidar_detection_ = detection;
        last_lidar_time_ = detection->header.stamp;
        last_lidar_receive_time_ = ros::WallTime::now();
        
        if (debug_mode_) {
            ROS_INFO_THROTTLE(1, "[Tracker] LiDAR detection: (%.3f, %.3f, %.3f)",
                detection->pose.position.x, detection->pose.position.y, detection->pose.position.z);
        }
    }

    void Tracker::camDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection)
    {
        std::lock_guard<std::mutex> lock(cam_mutex_);
        latest_cam_detection_ = detection;
        last_cam_time_ = detection->header.stamp;
        last_cam_receive_time_ = ros::WallTime::now();

        if (debug_mode_) {       
            ROS_INFO_THROTTLE(1, "[Tracker] Vision detection: (%.3f, %.3f, %.3f)",
                detection->pose.position.x, detection->pose.position.y, detection->pose.position.z);
        }
    }

    bool Tracker::transform2Odom(const std_msgs::Header& header, const Eigen::Vector3d& p_cam, Eigen::Vector3d& p_odom)
    {
        geometry_msgs::PointStamped point_in, point_out;
        point_in.header = header;
        point_in.point.x = p_cam.x();
        point_in.point.y = p_cam.y();
        point_in.point.z = p_cam.z();

        if (header.frame_id == target_frame_ || header.frame_id.empty()) {
            p_odom = p_cam;
            return true;
        }

        try 
        {
            // 降低阻塞超时时间，避免大于定时器周期造成积压
            geometry_msgs::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                target_frame_, header.frame_id, header.stamp, ros::Duration(0.005));
            tf2::doTransform(point_in, point_out, transform_stamped);
        } 
        catch (tf2::TransformException &ex) 
        {
            try 
            {
                // 异常回退使用的非阻塞查询
                geometry_msgs::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                    target_frame_, header.frame_id, ros::Time(0), ros::Duration(0.0));
                tf2::doTransform(point_in, point_out, transform_stamped);
            } 
            catch(tf2::TransformException &ex2) 
            {
                ROS_WARN_THROTTLE(1.0,
                                  "[Tracker] transform2Odom failed: %s -> %s at t=%.3f. err(stamped)=%s err(latest)=%s",
                                  header.frame_id.c_str(),
                                  target_frame_.c_str(),
                                  header.stamp.toSec(),
                                  ex.what(),
                                  ex2.what());
                return false;
            }
        }
        p_odom = Eigen::Vector3d(point_out.point.x, point_out.point.y, point_out.point.z);
        return true;
    }

    bool Tracker::updateProjectionExtrinsic(const std_msgs::Header& header)
    {
        if (header.frame_id.empty())
        {
            return false;
        }

        geometry_msgs::TransformStamped tf_world_to_cam;
        try
        {
            tf_world_to_cam = tf_buffer_->lookupTransform(
                header.frame_id, target_frame_, header.stamp, ros::Duration(0.005));
        }
        catch (tf2::TransformException&)
        {
            try
            {
                tf_world_to_cam = tf_buffer_->lookupTransform(
                    header.frame_id, target_frame_, ros::Time(0), ros::Duration(0.0));
            }
            catch (tf2::TransformException&)
            {
                return false;
            }
        }

        const auto& t = tf_world_to_cam.transform.translation;
        const auto& q = tf_world_to_cam.transform.rotation;
        Eigen::Quaterniond q_eigen(q.w, q.x, q.y, q.z);
        if (q_eigen.norm() < 1e-9)
        {
            return false;
        }
        q_eigen.normalize();

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = q_eigen.toRotationMatrix();
        T(0, 3) = t.x;
        T(1, 3) = t.y;
        T(2, 3) = t.z;

        std::lock_guard<std::mutex> lock(projection_mutex_);
        for (auto& model : loaded_models_)
        {
            if (model)
            {
                model->setProjectionExtrinsic(T);
            }
        }
        return true;
    }

    void Tracker::detectionCB(const ros::TimerEvent& event)
    {   
        (void)event;
        ros::Time now = ros::Time::now();

        rm_radar_msgs::DroneDetection::ConstPtr current_lidar_msg_ = nullptr;
        rm_radar_msgs::DroneDetection::ConstPtr current_cam_msg_ = nullptr;
        ros::WallTime cam_receive_time;
        ros::WallTime lidar_receive_time;

        {
            std::lock_guard<std::mutex> lock(cam_mutex_);
            if (latest_cam_detection_) {
                current_cam_msg_ = latest_cam_detection_;
                cam_receive_time = last_cam_receive_time_;
                latest_cam_detection_ = nullptr;
            }
        }
        {
            std::lock_guard<std::mutex> lock(lidar_mutex_);
            if (latest_lidar_detection_) {
                current_lidar_msg_ = latest_lidar_detection_;
                lidar_receive_time = last_lidar_receive_time_;
            }
        }

        const ros::WallTime now_wall = ros::WallTime::now();
        const double cam_age = cam_receive_time.isZero() ? std::numeric_limits<double>::infinity() : (now_wall - cam_receive_time).toSec();
        const double lidar_age = lidar_receive_time.isZero() ? std::numeric_limits<double>::infinity() : (now_wall - lidar_receive_time).toSec();

        if (last_time_.isZero()) {
            last_time_ = now;
            return;
        }

        double dt = (now - last_time_).toSec();
        if (dt < 0.001 || dt > 1.0) {
            last_time_ = now;
            return;
        }

        Eigen::VectorXd z_meas;
        Eigen::Vector3d z_init = Eigen::Vector3d::Zero();
        bool has_data = false;
        bool is_vision_active = false;
        bool is_cam_msg = false;
        bool is_lidar_msg = false;

        // Vision first
        if (current_cam_msg_ && cam_age < vision_timeout_)
        {
            if (current_cam_msg_->is_lidar_msg)
            {
                ROS_WARN_THROTTLE(1.0,
                                  "[Tracker] Drop cross-source msg on vision branch: is_lidar_msg=true (topic=/pose_solver/solved_pose)");
            }
            else
            {
                bool has_cam_info = false;
                std::array<double, 9> camera_k_local{};
                std::vector<double> camera_d_local;
                {
                    std::lock_guard<std::mutex> lock(projection_mutex_);
                    has_cam_info = has_camera_info_;
                    camera_k_local = camera_k_;
                    camera_d_local = camera_d_;
                }
                if (!has_cam_info)
                {
                    ROS_WARN_THROTTLE(1.0,
                                      "[Tracker] Waiting for CameraInfo on %s, skip vision measurement this frame.",
                                      cam_info_topic_.c_str());
                }

                Eigen::Vector3d p_cam(current_cam_msg_->pose.position.x, current_cam_msg_->pose.position.y, current_cam_msg_->pose.position.z);
                Eigen::Vector3d p_odom;
                updateProjectionExtrinsic(current_cam_msg_->header);

                if (has_cam_info && p_cam.norm() >= 0.01 && transform2Odom(current_cam_msg_->header, p_cam, p_odom))
                {
                    z_init = p_odom;
                    bool vision_measurement_valid = true;
                    bool has_4_corners = current_cam_msg_->armor_points.size() == 4;
                    if (force_3d_measurement_)
                    {
                        z_meas = z_init;
                    }
                    else if (has_4_corners)
                    {
                        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
                        for (int r = 0; r < 3; ++r)
                        {
                            for (int c = 0; c < 3; ++c)
                            {
                                K.at<double>(r, c) = camera_k_local[r * 3 + c];
                            }
                        }
                        cv::Mat D;
                        if (!camera_d_local.empty())
                        {
                            D = cv::Mat::zeros(static_cast<int>(camera_d_local.size()), 1, CV_64F);
                            for (size_t i = 0; i < camera_d_local.size(); ++i)
                            {
                                D.at<double>(static_cast<int>(i), 0) = camera_d_local[i];
                            }
                        }

                        std::vector<cv::Point2f> raw_points(4);
                        for (int i = 0; i < 4; ++i)
                        {
                            raw_points[i] = cv::Point2f(
                                static_cast<float>(current_cam_msg_->armor_points[i].x),
                                static_cast<float>(current_cam_msg_->armor_points[i].y));
                        }

                        std::vector<cv::Point2f> undistorted_points;
                        cv::undistortPoints(raw_points, undistorted_points, K, D, cv::noArray(), K);
                        if (undistorted_points.size() != 4)
                        {
                            ROS_WARN_THROTTLE(1.0, "[Tracker] Undistort failed, skip vision measurement this frame.");
                            vision_measurement_valid = false;
                        }
                        else
                        {
                            z_meas = Eigen::VectorXd::Zero(8);
                            for (int i = 0; i < 4; ++i)
                            {
                                z_meas(2 * i) = undistorted_points[i].x;
                                z_meas(2 * i + 1) = undistorted_points[i].y;
                            }
                        }
                    }
                    else
                    {
                        if (enable_3d_fallback_)
                        {
                            z_meas = z_init;
                        }
                        else
                        {
                            vision_measurement_valid = false;
                            ROS_WARN_THROTTLE(1.0,
                                              "[Tracker] Strict 8D mode: armor corners != 4, skip vision update.");
                        }
                    }
                    has_data = vision_measurement_valid;
                    is_vision_active = vision_measurement_valid;
                    if (vision_measurement_valid)
                    {
                        is_cam_msg = true;
                        is_lidar_msg = false;
                        current_time_ = current_cam_msg_->header.stamp;
                    }

                    tf2::Quaternion q_new(
                        current_cam_msg_->pose.orientation.x, current_cam_msg_->pose.orientation.y,
                        current_cam_msg_->pose.orientation.z, current_cam_msg_->pose.orientation.w);
                    if (q_new.length() > 0.001 && q_new.length() < 2.0)
                    {
                        q_new.normalize();
                        if (q_track_valid_) {
                            double dot = q_track_.x() * q_new.x() + q_track_.y() * q_new.y() +
                                         q_track_.z() * q_new.z() + q_track_.w() * q_new.w();
                            if (dot < 0) q_new = tf2::Quaternion(-q_new.x(), -q_new.y(), -q_new.z(), -q_new.w());
                        }
                        q_track_ = q_new;
                        q_track_valid_ = true;
                    }
                }
            }
        }

        // LiDAR fallback
        if (!has_data && use_lidar_fusion_ && current_lidar_msg_ && lidar_age < lidar_timeout_)
        {
            Eigen::Vector3d p_lidar(current_lidar_msg_->pose.position.x, current_lidar_msg_->pose.position.y, current_lidar_msg_->pose.position.z);
            Eigen::Vector3d p_odom;

            if (transform2Odom(current_lidar_msg_->header, p_lidar, p_odom))
            {
                z_meas = p_odom;
                z_init = p_odom;
                has_data = true;
                is_lidar_msg = current_lidar_msg_->is_lidar_msg || !current_lidar_msg_->is_cam_msg;
                is_cam_msg = current_lidar_msg_->is_cam_msg && !is_lidar_msg;
                current_time_ = current_lidar_msg_->header.stamp;
            }
        }

        // Adjust measurement noise based on source
        if (has_data && filterIsInitialized()) 
        {
            Eigen::MatrixXd R_base = filterGetBaseMeasurementNoise();
            
            ///test
            if (z_meas.size() == 8)
            {
                Eigen::MatrixXd R_pixel = Eigen::MatrixXd::Zero(8, 8);
                const double u_var = pixel_noise_u_ * pixel_noise_u_;
                const double v_var = pixel_noise_v_ * pixel_noise_v_;
                for (int i = 0; i < 4; ++i)
                {
                    R_pixel(2 * i, 2 * i) = u_var;
                    R_pixel(2 * i + 1, 2 * i + 1) = v_var;
                }
                filterSetMeasurementNoise(R_pixel);
            }
            else
            {
                Eigen::Matrix3d R_matrix = Eigen::Matrix3d::Zero();
                if (R_base.rows() < 3 || R_base.cols() < 3)
                {
                    R_base = Eigen::MatrixXd::Identity(3, 3);
                }

                if (is_vision_active)
                {
                    R_matrix(0, 0) = R_base(0, 0) * 1.0;
                    R_matrix(1, 1) = R_base(1, 1) * 1.0;
                    R_matrix(2, 2) = R_base(2, 2) * 10.0;
                }
                else
                {
                    R_matrix(0, 0) = R_base(0, 0) * 5.0;
                    R_matrix(1, 1) = R_base(1, 1) * 5.0;
                    R_matrix(2, 2) = R_base(2, 2) * 0.1;
                }
                filterSetMeasurementNoise(R_matrix);
            }
        }

        if (!has_data) {
            current_time_ = current_time_ + ros::Duration(dt); // 如果没有新数据，继续推进时间以触发预测和状态转移，+dt保持时间轴的物理连续性
        }
        else if (enable_pixel_gating_ && z_meas.size() == 8 && track_state_ != rm_track::LOST)
        {
            double nis = 0.0;
            double rmse = 0.0;
            double rmse_gate = 0.0;
            const double distance = z_init.norm();
            if (!validatePixelMeasurement(z_meas, distance, nis, rmse, rmse_gate))
            {
                if (enable_3d_fallback_)
                {
                    // Fallback to 3D pose update when 8D pixel gate fails.
                    z_meas = z_init;
                    has_data = true;

                    Eigen::MatrixXd R_base = filterGetBaseMeasurementNoise();
                    if (R_base.rows() < 3 || R_base.cols() < 3)
                    {
                        R_base = Eigen::MatrixXd::Identity(3, 3);
                    }
                    Eigen::Matrix3d R_matrix = Eigen::Matrix3d::Zero();
                    R_matrix(0, 0) = R_base(0, 0) * 1.0;
                    R_matrix(1, 1) = R_base(1, 1) * 1.0;
                    R_matrix(2, 2) = R_base(2, 2) * 10.0;
                    filterSetMeasurementNoise(R_matrix);

                    ROS_WARN_THROTTLE(1.0,
                                      "[Tracker] Pixel gated out -> fallback to 3D: dist=%.2f m, NIS=%.3f (cfg=%.3f), RMSE=%.3fpx (active=%.3f), cfg_rmse[n/m/f]=%.3f/%.3f/%.3f, cfg_dist[n/f]=%.2f/%.2f",
                                      distance, nis, pixel_nis_gate_, rmse, rmse_gate,
                                      pixel_reproj_rmse_gate_near_, pixel_reproj_rmse_gate_mid_, pixel_reproj_rmse_gate_far_,
                                      pixel_gate_distance_near_, pixel_gate_distance_far_);
                }
                else
                {
                    has_data = false;
                    ROS_WARN_THROTTLE(1.0,
                                      "[Tracker] Strict 8D mode: pixel gated out, skip update. dist=%.2f m, NIS=%.3f (cfg=%.3f), RMSE=%.3fpx (active=%.3f)",
                                      distance, nis, pixel_nis_gate_, rmse, rmse_gate);
                }
            }
        }
        
        processTracking(has_data, is_cam_msg, is_lidar_msg, z_meas, z_init, dt);
        last_time_ = now;
    }

    bool Tracker::filterIsInitialized() const {
        return use_aimm_ ? (aimm_ && aimm_->isInitialized()) : (ukf_ && ukf_->isInitialized());
    }

    void Tracker::filterSetDt(double dt) {
        if (use_aimm_) { if (aimm_) aimm_->setDt(dt); }
        else { if (ukf_) ukf_->setDt(dt); }
    }

    void Tracker::filterPredict() {
        if (use_aimm_) { if (aimm_) aimm_->predict(); }
        else { if (ukf_) ukf_->predict(); }
    }

    void Tracker::filterUpdate(const Eigen::VectorXd& z_meas) {
        if (use_aimm_) { if (aimm_) aimm_->update(z_meas); }
        else { if (ukf_) ukf_->update(z_meas); }
    }

    Eigen::VectorXd Tracker::filterGetState() const
    {
        if (use_aimm_ && aimm_)
        {
            return aimm_->getState();
        }
        else if(ukf_)
        {
            ModelState raw_state = {ukf_->getState(), ukf_->getCovariance()};
            return ukf_->getModel()->toUnifiedState(raw_state).state;
        }
        return Eigen::VectorXd::Zero(9);
    }

    Eigen::MatrixXd Tracker::filterGetBaseMeasurementNoise() const {
        if (use_aimm_ && aimm_) return aimm_->getBaseMeasurementNoise();
        if (!use_aimm_ && ukf_) return ukf_->getBaseMeasurementNoise();
        return Eigen::MatrixXd::Identity(3,3); 
    }

    void Tracker::filterSetMeasurementNoise(const Eigen::MatrixXd& R) {
        if (use_aimm_) { if (aimm_) aimm_->setMeasurementNoise(R); }
        else { if (ukf_) ukf_->setMeasurementNoise(R); }
    }

    void Tracker::handleDetectingState(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt)
    {
        (void)z_init;
        (void)dt;
        if (has_data) {
            detect_fail_count_ = 0;  // Reset continuous failure count

            if (is_lidar_msg && !is_cam_msg) {
                lidar_detected_count_++;
                cam_detected_count_ = 0;
                detected_count_ = lidar_detected_count_;
                tracking_with_cam_ = false;
            } else {
                cam_detected_count_++;
                lidar_detected_count_ = 0;
                detected_count_ = cam_detected_count_;
                tracking_with_cam_ = true;
            }

            filterUpdate(z_meas);
            const int active_hit_threshold = tracking_with_cam_ ? cam_hitcount_ : lidar_hitcount_;
            if(detected_count_ >= active_hit_threshold) {
                track_state_ = rm_track::TRACKING;
                is_tracking_ = true; 
                detected_count_ = 0; 
                cam_detected_count_ = 0;
                lidar_detected_count_ = 0;
                cam_lost_counter_ = 0;
                lidar_lost_counter_ = 0;
                ROS_WARN("[Tracker] STATE: DETECTING -> TRACKING");
            }
        } 
        else 
        {   
            detect_fail_count_++;
            if (detect_fail_count_ > max_lost_count_) {
                track_state_ = rm_track::LOST;
                detect_fail_count_ = 0;
                q_track_valid_ = false;
                ROS_WARN("[Tracker] STATE: DETECTING -> LOST");
            }
        }
    }

    void Tracker::handleTrackingState(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt)
    {
        (void)z_init;
        (void)dt;
        if (has_data) {
            lost_count_ = 0;
            if (is_lidar_msg && !is_cam_msg) {
                tracking_with_cam_ = false;
                lidar_lost_counter_ = 0;
            } else {
                tracking_with_cam_ = true;
                cam_lost_counter_ = 0;
            }
            filterUpdate(z_meas);
            is_tracking_ = true;
        } 
        else 
        {
            track_state_ = rm_track::TEMP_LOST;
            if (tracking_with_cam_) {
                cam_lost_counter_ = 1;
                lost_count_ = cam_lost_counter_;
            } else {
                lidar_lost_counter_ = 1;
                lost_count_ = lidar_lost_counter_;
            }
            is_tracking_ = false;
            ROS_WARN("[Tracker] STATE: TRACKING -> TEMP_LOST");
        }
    }

    void Tracker::handleTempLostState(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt)
    {
        (void)z_init;
        (void)dt;
        if (has_data) 
        {
            track_state_ = rm_track::TRACKING;
            is_tracking_ = true; 
            if (is_lidar_msg && !is_cam_msg) {
                tracking_with_cam_ = false;
                lidar_lost_counter_ = 0;
            } else {
                tracking_with_cam_ = true;
                cam_lost_counter_ = 0;
            }
            lost_count_ = 0;
            filterUpdate(z_meas);
            ROS_WARN("[Tracker] STATE: TEMP_LOST -> TRACKING");
        } 
        else 
        {
            const int active_lost_threshold = tracking_with_cam_ ? cam_lostcount_ : lidar_lostcount_;
            if (tracking_with_cam_) {
                cam_lost_counter_++;
                lost_count_ = cam_lost_counter_;
            } else {
                lidar_lost_counter_++;
                lost_count_ = lidar_lost_counter_;
            }
            if(lost_count_ > active_lost_threshold) 
            {
                track_state_ = rm_track::LOST;
                is_tracking_ = false;
                q_track_valid_ = false; 
                cam_lost_counter_ = 0;
                lidar_lost_counter_ = 0;
                ROS_WARN("[Tracker] STATE: TEMP_LOST -> LOST");
                
                // When entering LOST state, ensure we have the last known position
                if (filterIsInitialized()) {
                    Eigen::VectorXd state = filterGetState();
                    if (state.size() >= 6) {
                        last_known_position_ << state(0), state(1), state(2);
                        last_known_velocity_ << state(3), state(4), state(5);
                    }
                }
            } 
        }
    }

    void Tracker::handleLostState(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt)
    {
        (void)dt;
        if (has_data) 
        {
            track_state_ = rm_track::DETECTING;
            if (is_lidar_msg && !is_cam_msg) {
                tracking_with_cam_ = false;
                lidar_detected_count_ = 1;
                cam_detected_count_ = 0;
                detected_count_ = lidar_detected_count_;
            } else {
                tracking_with_cam_ = true;
                cam_detected_count_ = 1;
                lidar_detected_count_ = 0;
                detected_count_ = cam_detected_count_;
            }
            detect_fail_count_ = 0;
            lost_count_ = 0;
            cam_lost_counter_ = 0;
            lidar_lost_counter_ = 0;

            if (use_aimm_ && aimm_) {
                aimm_->initialize(z_init);
                ROS_WARN("[Tracker] STATE: LOST -> DETECTING (AIMM)");
            } else if (!use_aimm_ && ukf_) {
                // 在初始化前检查UKF及其模型是否有效
                if (ukf_ && ukf_->getModel()) {
                    ukf_->initialize(z_init);
                        ROS_WARN("[Tracker] STATE: LOST -> DETECTING (UKF)");
                } else {
                    ROS_ERROR("[Tracker] Cannot initialize UKF: ukf or model is null");
                }
            }
        }
    }

    void Tracker::processTracking(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt)
    {
        if(track_state_ != rm_track::LOST && (dt < 0.0001 || dt > 1.0)) return;

        // Predict continuously while tracking lifecycle is active, even without measurements.
        if (track_state_ != rm_track::LOST && filterIsInitialized()) {
            filterSetDt(dt);
            filterPredict();
        }

        switch (track_state_) {
            case rm_track::DETECTING: handleDetectingState(has_data, is_cam_msg, is_lidar_msg, z_meas, z_init, dt); break;
            case rm_track::TRACKING:  handleTrackingState(has_data, is_cam_msg, is_lidar_msg, z_meas, z_init, dt);  break;
            case rm_track::TEMP_LOST: handleTempLostState(has_data, is_cam_msg, is_lidar_msg, z_meas, z_init, dt);  break;
            case rm_track::LOST:      handleLostState(has_data, is_cam_msg, is_lidar_msg, z_meas, z_init, dt);      break;
            default: break;
        }
        if (state_log_mode_) {
            ROS_WARN_THROTTLE(1, "[Tracker] State: %s, Src: %s, Has Data: %s, Hit(c/l): %d/%d, Lost(c/l): %d/%d", 
                track_state_ == rm_track::DETECTING ? "DETECTING" : 
                track_state_ == rm_track::TRACKING ? "TRACKING" : 
                track_state_ == rm_track::TEMP_LOST ? "TEMP_LOST" : "LOST",
                tracking_with_cam_ ? "CAM" : "LIDAR",
                has_data ? "YES" : "NO", cam_detected_count_, lidar_detected_count_, cam_lost_counter_, lidar_lost_counter_);
        }

        // Always publish tracker data regardless of state to let image processor know the current status
        publishTrackerData();
    }

    void Tracker::publishTrackerData()
    {
        if (use_aimm_ && aimm_debug_mode_ && aimm_ && aimm_->isInitialized())
        {
            rm_radar_msgs::aimm_debugger aimm_msg = aimm_->getDebugMsg();
            aimm_msg.header.stamp = ros::Time::now();
            aimm_msg.header.frame_id = target_frame_;

            aimm_debug_pub_.publish(aimm_msg);
        }

        rm_msgs::TrackData track_data_msg;
        track_data_msg.header.stamp = current_time_;
        track_data_msg.header.frame_id = target_frame_;

        track_data_msg.rotation.w = q_track_.getW();
        track_data_msg.rotation.x = q_track_.getX();
        track_data_msg.rotation.y = q_track_.getY();
        track_data_msg.rotation.z = q_track_.getZ();

        track_data_msg.tracking = is_tracking_;

        // Even if filter is not initialized, still publish the tracking status
        if(filterIsInitialized()) {
            Eigen::VectorXd state = filterGetState();

            if(state.size() >= 6) {
                track_data_msg.position.x = state(0);
                track_data_msg.position.y = state(1);
                track_data_msg.position.z = state(2);
                track_data_msg.velocity.x = state(3);
                track_data_msg.velocity.y = state(4);
                track_data_msg.velocity.z = state(5);
                
                // Update last known position and velocity when we have valid data
                last_known_position_ << state(0), state(1), state(2);
                last_known_velocity_ << state(3), state(4), state(5);
            }
        } else {
            // Use last known position and velocity when filter is not initialized
            track_data_msg.position.x = last_known_position_(0);
            track_data_msg.position.y = last_known_position_(1);
            track_data_msg.position.z = last_known_position_(2);
            track_data_msg.velocity.x = last_known_velocity_(0);
            track_data_msg.velocity.y = last_known_velocity_(1);
            track_data_msg.velocity.z = last_known_velocity_(2);
        }
        
        tracker_pub_.publish(track_data_msg);

        if (q_track_valid_) {
            geometry_msgs::TransformStamped track_tf;
            track_tf.header.stamp = current_time_;
            track_tf.header.frame_id = target_frame_;
            track_tf.child_frame_id = "tracked_target_" ;
            track_tf.transform.translation.x = track_data_msg.position.x;
            track_tf.transform.translation.y = track_data_msg.position.y;
            track_tf.transform.translation.z = track_data_msg.position.z;
            track_tf.transform.rotation.w = track_data_msg.rotation.w;
            track_tf.transform.rotation.x = track_data_msg.rotation.x;
            track_tf.transform.rotation.y = track_data_msg.rotation.y;
            track_tf.transform.rotation.z = track_data_msg.rotation.z;
            tf_broadcaster_->sendTransform(track_tf);
        }
    }
}