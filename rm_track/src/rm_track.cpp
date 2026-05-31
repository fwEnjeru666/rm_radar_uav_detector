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
            ROS_WARN("[Tracker] No pure UKF models loaded. Falling back to model index 0.");
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

    void Tracker::applyProjectionParamsToModels(const CameraIntrinsics& intrinsics)
    {
        std::lock_guard<std::mutex> lock(projection_mutex_);
        for (auto& model : loaded_models_) {
            if (!model) continue;
            model->setProjectionParams(intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                                       projection_params_.armor_width, projection_params_.armor_height);
        }
    }

    void Tracker::camInfoCB(const sensor_msgs::CameraInfoConstPtr& cam_info) 
    {
        if (!cam_info || cam_info->K[0] <= 1e-6) return;
        std::lock_guard<std::mutex> lock(projection_mutex_);
        std::string frame_id = cam_info->header.frame_id;
        CameraIntrinsics& intrinsics = camera_intrinsics_map_[frame_id];
        intrinsics.fx = cam_info->K[0]; intrinsics.fy = cam_info->K[4];
        intrinsics.cx = cam_info->K[2]; intrinsics.cy = cam_info->K[5];
        for (size_t i = 0; i < 9; ++i) intrinsics.K[i] = cam_info->K[i];
        intrinsics.D = cam_info->D;
    }

    void Tracker::teleDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection)
    {
        std::lock_guard<std::mutex> lock(tele_cam_mutex_);
        latest_tele_cam_detection_ = detection;
        last_tele_receive_time_ = ros::WallTime::now();
    }

    void Tracker::wideDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection)
    {
        std::lock_guard<std::mutex> lock(wide_cam_mutex_);
        latest_wide_cam_detection_ = detection;
        last_wide_receive_time_ = ros::WallTime::now();
    }

    bool Tracker::processVisionMeasurement(const rm_radar_msgs::DroneDetection::ConstPtr& cam_msg, 
                                           Eigen::VectorXd& z_meas, Eigen::Vector3d& z_init, std::string& active_frame_id) 
    {
        if (cam_msg->is_lidar_msg) return false;
        active_frame_id = cam_msg->header.frame_id;
        CameraIntrinsics current_intrinsics;
        {
            std::lock_guard<std::mutex> lock(projection_mutex_);
            if (camera_intrinsics_map_.find(active_frame_id) == camera_intrinsics_map_.end()) return false;
            current_intrinsics = camera_intrinsics_map_[active_frame_id];
        }

        if (active_frame_id != last_active_frame_id_) 
        {
            applyProjectionParamsToModels(current_intrinsics);
            last_active_frame_id_ = active_frame_id;
        }

        Eigen::Vector3d p_cam(cam_msg->pose.position.x, cam_msg->pose.position.y, cam_msg->pose.position.z);
        Eigen::Vector3d p_odom;
        
        // TF将相机坐标系下的测量转换至 odom 坐标系
        if (p_cam.norm() >= 0.01 && transform2Odom(cam_msg->header, p_cam, p_odom)) 
        {
            z_init = p_odom;
            if (force_3d_measurement_) { 
                z_meas = z_init; 
            } 
            else if (cam_msg->armor_points.size() == 4) {
                cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
                for (int r=0; r<3; ++r) for(int c=0; c<3; ++c) K.at<double>(r,c) = current_intrinsics.K[r*3+c];
                cv::Mat D = cv::Mat::zeros(current_intrinsics.D.size(), 1, CV_64F);
                for (size_t i=0; i<current_intrinsics.D.size(); ++i) D.at<double>(i,0) = current_intrinsics.D[i];
                std::vector<cv::Point2f> raw_pts(4), undist_pts;
                for (int i=0; i<4; ++i) raw_pts[i] = cv::Point2f(cam_msg->armor_points[i].x, cam_msg->armor_points[i].y);
                cv::undistortPoints(raw_pts, undist_pts, K, D, cv::noArray(), K);
                if (undist_pts.size() == 4) {
                    z_meas = Eigen::VectorXd::Zero(8);
                    for (int i=0; i<4; ++i) { z_meas(2*i) = undist_pts[i].x; z_meas(2*i+1) = undist_pts[i].y; }
                } else return false;
            } else return false;

            tf2::Quaternion q_new(cam_msg->pose.orientation.x, cam_msg->pose.orientation.y, cam_msg->pose.orientation.z, cam_msg->pose.orientation.w);
            if (q_new.length() > 0.001) 
            {
                q_new.normalize();
                if (q_track_valid_ && (q_track_.x()*q_new.x() + q_track_.y()*q_new.y() + q_track_.z()*q_new.z() + q_track_.w()*q_new.w() < 0)) 
                    q_new = tf2::Quaternion(-q_new.x(), -q_new.y(), -q_new.z(), -q_new.w());
                q_track_ = q_new; q_track_valid_ = true;
            }

            // 紧耦合更新外参
            if (!force_3d_measurement_) 
            {
                getProjectionExtrinsic(cam_msg->header);
            }

            return true;
        }
        return false;
    }

    void Tracker::onInit()
    {
        ros::NodeHandle nh = getMTPrivateNodeHandle();
        nh_ = nh;
        ROS_INFO("[Tracker] Initializing nodelet...");
        ROS_INFO("[Tracker] Nodelet name: %s, private namespace: %s",
                 getName().c_str(), nh_.getNamespace().c_str());

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
        tele_pixel_noise_u_ = nh_.param("tele_pixel_noise_u", 4.0);
        tele_pixel_noise_v_ = nh_.param("tele_pixel_noise_v", 4.0);
        wide_pixel_noise_u_ = nh_.param("wide_pixel_noise_u", 4.0);
        wide_pixel_noise_v_ = nh_.param("wide_pixel_noise_v", 4.0);
        force_3d_measurement_ = nh_.param("force_3d_measurement", false);
        enable_pixel_gating_ = nh_.param("enable_pixel_gating", true);
        enable_3d_fallback_ = nh_.param("enable_3d_fallback", true);
        pixel_nis_gate_ = nh_.param("pixel_nis_gate", 40.0);
        pixel_reproj_rmse_gate_near_ = nh_.param("pixel_reproj_rmse_gate_near", 12.0);
        pixel_reproj_rmse_gate_mid_ = nh_.param("pixel_reproj_rmse_gate_mid", 20.0);
        pixel_reproj_rmse_gate_far_ = nh_.param("pixel_reproj_rmse_gate_far", 30.0);
        pixel_gate_distance_near_ = nh_.param("pixel_gate_distance_near", 8.0);
        pixel_gate_distance_far_ = nh_.param("pixel_gate_distance_far", 18.0);

        tele_cam_info_sub_ = nh_.subscribe("/tele_camera/camera_info", 1, &Tracker::camInfoCB, this);
        tele_detection_sub_ = nh_.subscribe("/tele_camera/detection", 10, &Tracker::teleDetectionCB, this);
        wide_cam_info_sub_ = nh_.subscribe("/wide_camera/camera_info", 1, &Tracker::camInfoCB, this);
        wide_detection_sub_ = nh_.subscribe("/wide_camera/detection", 10, &Tracker::wideDetectionCB, this);
        lidar_detection_sub_ = nh_.subscribe("/lidar_detector/lidar_detection", 1, &Tracker::lidarDetectionCB, this);
        tracker_pub_ = nh_.advertise<rm_msgs::GimbalCmd>("/tracker/track_data", 1);
        img_tracker_pub_ = nh_.advertise<rm_msgs::TrackData>("/tracker/img_track_data", 1);
        aimm_debug_pub_ = nh_.advertise<rm_radar_msgs::aimm_debugger>("/tracker/aimm_debug", 1);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(ros::Duration(10));
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ =  std::make_shared<tf2_ros::TransformBroadcaster>();

        detection_timer_ = nh_.createTimer(ros::Duration(0.02), &Tracker::detectionCB, this);  

        model_manager_ = std::make_shared<model_register>(nh_);
        if (model_manager_->load_models() == 0)
        {
            ROS_FATAL("[Tracker] No models loaded. Shutting down.");
            ros::shutdown();
            return;
        }
        loaded_models_ = model_manager_->get_all_models();
        loaded_model_names_ = model_manager_->get_model_names();

        aimm_ = std::make_shared<AIMM>(nh_, loaded_models_);
        ROS_INFO("[Tracker] Initialized AIMM with %lu models.", loaded_models_.size());

        pure_ukf_model_selector_ = nh_.param("pure_ukf_model_type", 0);
        if (loaded_models_.empty()) 
        {
            ROS_ERROR("[Tracker] No models loaded. Cannot initialize UKF.");
            ros::shutdown();
            return;
        }
        pure_ukf_model_idx_ = resolvePureUkfModelIndex(pure_ukf_model_selector_, true);

        ros::NodeHandle pure_ukf_nh(nh_, "pure_ukf");
        ukf_ = std::make_shared<UKF>(pure_ukf_nh, loaded_models_[pure_ukf_model_idx_]);

        track_cfg_srv_.reset(new dynamic_reconfigure::Server<rm_track::trackConfig>(nh_));
        track_cfg_cb_ = boost::bind(&Tracker::trackconfigCB, this, _1, _2);
        track_cfg_srv_->setCallback(track_cfg_cb_);
        ROS_INFO("[Tracker] Dynamic reconfigure attached at namespace: %s", nh_.getNamespace().c_str());

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
        tele_pixel_noise_u_ = config.tele_pixel_noise_u;
        tele_pixel_noise_v_ = config.tele_pixel_noise_v;
        wide_pixel_noise_u_ = config.wide_pixel_noise_u;
        wide_pixel_noise_v_ = config.wide_pixel_noise_v;
        getPixelMeasurementNoise();

        tele_r_pos_xy_ = config.tele_r_pos_xy;
        tele_r_pos_z_ = config.tele_r_pos_z;
        tele_r_yaw_ = config.tele_r_yaw;
        wide_r_pos_xy_ = config.wide_r_pos_xy;
        wide_r_pos_z_ = config.wide_r_pos_z;
        wide_r_yaw_ = config.wide_r_yaw;
        lidar_r_pos_xy_ = config.lidar_r_pos_xy;
        lidar_r_pos_z_ = config.lidar_r_pos_z;
        lidar_r_yaw_ = config.lidar_r_yaw;
        get3DMeasurementNoise();

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

        if (config.use_aimm != use_aimm_) {
            use_aimm_ = config.use_aimm;
            needs_reset = true;
        }

        if (config.pure_ukf_model_type != pure_ukf_model_selector_) {
            pure_ukf_model_selector_ = config.pure_ukf_model_type;
            const int resolved_idx = resolvePureUkfModelIndex(pure_ukf_model_selector_, true);
            if (resolved_idx != pure_ukf_model_idx_) {
                pure_ukf_model_idx_ = resolved_idx;
                if (!use_aimm_) {
                    ukf_->setModel(loaded_models_[pure_ukf_model_idx_]);
                    needs_reset = true;
                }
            }
        }

        if (needs_reset) 
        {
            track_state_ = rm_track::LOST;
            is_tracking_ = false;
            q_track_valid_ = false;
        }

        if (use_aimm_) 
        {
            log_msg << "Estimator: AIMM.";
        } else 
        {
            log_msg << "Estimator: UKF, Model: " << loaded_model_names_[pure_ukf_model_idx_] << ".";
        }
        if (needs_reset) 
        {
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
        
        if (debug_mode_) 
        {
            ROS_INFO_THROTTLE(1, "[Tracker] LiDAR detection: (%.3f, %.3f, %.3f)",
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
            geometry_msgs::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                target_frame_, header.frame_id, header.stamp, ros::Duration(0.005));
            tf2::doTransform(point_in, point_out, transform_stamped);
            if(debug_mode_) {
                ROS_INFO_THROTTLE(1, "[Tracker] TF transform success: %s -> %s at t=%.3f",
                    header.frame_id.c_str(), target_frame_.c_str(), header.stamp.toSec());
            }
        } 
        catch (tf2::TransformException &ex) 
        {
            try 
            {
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

    bool Tracker::getProjectionExtrinsic(const std_msgs::Header& header)
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

    void Tracker::getPixelMeasurementNoise() 
    { 
        std::lock_guard<std::mutex> lock(noise_mtx_);
        R_tele_pixel_.setZero(8, 8); 
        R_wide_pixel_.setZero(8, 8); 
        
        double u_var_tele = tele_pixel_noise_u_ * tele_pixel_noise_u_; 
        double v_var_tele = tele_pixel_noise_v_ * tele_pixel_noise_v_; 
        double u_var_wide = wide_pixel_noise_u_ * wide_pixel_noise_u_; 
        double v_var_wide = wide_pixel_noise_v_ * wide_pixel_noise_v_; 
        
        for (int i = 0; i < 4; ++i) {
            R_tele_pixel_(2*i, 2*i) = u_var_tele;      
            R_tele_pixel_(2*i+1, 2*i+1) = v_var_tele;  
            R_wide_pixel_(2*i, 2*i) = u_var_wide;     
            R_wide_pixel_(2*i+1, 2*i+1) = v_var_wide;  
        }
    }

    void Tracker::get3DMeasurementNoise() 
    { 
        std::lock_guard<std::mutex> lock(noise_mtx_);
        R_tele_.setZero(3, 3); 
        R_wide_.setZero(3, 3); 
        R_lidar_.setZero(3, 3); 

        R_tele_(0, 0) = tele_r_pos_xy_ * tele_r_pos_xy_; 
        R_tele_(1, 1) = tele_r_pos_xy_ * tele_r_pos_xy_; 
        R_tele_(2, 2) = tele_r_pos_z_ * tele_r_pos_z_; 

        R_wide_(0, 0) = wide_r_pos_xy_ * wide_r_pos_xy_; 
        R_wide_(1, 1) = wide_r_pos_xy_ * wide_r_pos_xy_; 
        R_wide_(2, 2) = wide_r_pos_z_ * wide_r_pos_z_; 

        R_lidar_(0, 0) = lidar_r_pos_xy_ * lidar_r_pos_xy_; 
        R_lidar_(1, 1) = lidar_r_pos_xy_ * lidar_r_pos_xy_; 
        R_lidar_(2, 2) = lidar_r_pos_z_ * lidar_r_pos_z_; 
    }


    void Tracker::detectionCB(const ros::TimerEvent& event) 
    {   
        ros::Time now = ros::Time::now();
        rm_radar_msgs::DroneDetection::ConstPtr msg_tele = nullptr, msg_wide = nullptr, msg_lidar = nullptr;
        ros::WallTime t_tele, t_wide, t_lidar;

        { std::lock_guard<std::mutex> lock(tele_cam_mutex_); if (latest_tele_cam_detection_) { msg_tele = latest_tele_cam_detection_; t_tele = last_tele_receive_time_; latest_tele_cam_detection_ = nullptr; } }
        { std::lock_guard<std::mutex> lock(wide_cam_mutex_); if (latest_wide_cam_detection_) { msg_wide = latest_wide_cam_detection_; t_wide = last_wide_receive_time_; latest_wide_cam_detection_ = nullptr; } }
        { std::lock_guard<std::mutex> lock(lidar_mutex_); if (latest_lidar_detection_) { msg_lidar = latest_lidar_detection_; t_lidar = last_lidar_receive_time_; latest_lidar_detection_ = nullptr; } }

        ros::WallTime now_wall = ros::WallTime::now();
        double age_tele = t_tele.isZero() ? 999.0 : (now_wall - t_tele).toSec();
        double age_wide = t_wide.isZero() ? 999.0 : (now_wall - t_wide).toSec();
        double age_lidar = t_lidar.isZero() ? 999.0 : (now_wall - t_lidar).toSec();

        if (last_time_.isZero() || (now - last_time_).toSec() < 0.001 || (now - last_time_).toSec() > 1.0) { last_time_ = now; return; }
        double dt = (now - last_time_).toSec();

        Eigen::VectorXd z_meas; Eigen::Vector3d z_init = Eigen::Vector3d::Zero();
        bool has_data = false, is_cam_msg = false, is_lidar_msg = false;
        std::string active_frame_id = "";

        if (msg_tele && age_tele < vision_timeout_) {
            if (processVisionMeasurement(msg_tele, z_meas, z_init, active_frame_id)) {
                has_data = true; is_cam_msg = true; current_time_ = msg_tele->header.stamp;
            }
        }
        if (!has_data && msg_wide && age_wide < vision_timeout_) {
            if (processVisionMeasurement(msg_wide, z_meas, z_init, active_frame_id)) {
                has_data = true; is_cam_msg = true; current_time_ = msg_wide->header.stamp;
            }
        }
        if (!has_data && use_lidar_fusion_ && msg_lidar && age_lidar < lidar_timeout_) {
            Eigen::Vector3d p_lidar(msg_lidar->pose.position.x, msg_lidar->pose.position.y, msg_lidar->pose.position.z);
            Eigen::Vector3d p_odom;
            if (transform2Odom(msg_lidar->header, p_lidar, p_odom)) {
                active_frame_id = msg_lidar->header.frame_id;
                z_meas = p_odom; 
                z_init = p_odom; 
                has_data = true; 
                is_lidar_msg = true; 
                current_time_ = msg_lidar->header.stamp;
            } else {
                ROS_WARN_THROTTLE(1.0, "[Tracker] Failed to transform Lidar detection to odom!");
            }
        }

        if (has_data && filterIsInitialized()) 
        {
            if (z_meas.size() == 8) 
            {
                std::lock_guard<std::mutex> lock(noise_mtx_);
                if(active_frame_id.find("left") != std::string::npos) 
                {
                    filterSetMeasurementNoise(R_tele_pixel_);
                } 
                else if(active_frame_id.find("right") != std::string::npos) 
                {
                    filterSetMeasurementNoise(R_wide_pixel_);
                } 
                else 
                {
                    Eigen::MatrixXd R_pixel = Eigen::MatrixXd::Identity(8, 8);
                    filterSetMeasurementNoise(R_pixel);
                    ROS_WARN("R_pixel set to identity due to unknown camera frame_id: %s", active_frame_id.c_str()); 
                }
            } 
            else 
            {
                std::lock_guard<std::mutex> lock(noise_mtx_);
                if(is_lidar_msg)
                {
                    filterSetMeasurementNoise(R_lidar_);
                }
                else if(active_frame_id.find("left") != std::string::npos) 
                {
                    filterSetMeasurementNoise(R_tele_);
                } 
                else if(active_frame_id.find("right") != std::string::npos) 
                {
                    filterSetMeasurementNoise(R_wide_);
                } 
                else
                {
                    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3);
                    filterSetMeasurementNoise(R);
                    ROS_WARN("R_3d set to identity due to unknown camera frame_id: %s", active_frame_id.c_str()); 
                }
            }
        }

        if (!has_data) current_time_ = current_time_ + ros::Duration(dt);
        processTracking(has_data, is_cam_msg, is_lidar_msg, z_meas, z_init, dt);
        last_time_ = now;
    }
    bool Tracker::filterIsInitialized() const 
    {
        return use_aimm_ ? (aimm_ && aimm_->isInitialized()) : (ukf_ && ukf_->isInitialized());
    }

    void Tracker::filterSetDt(double dt) 
    {
        if (use_aimm_) { if (aimm_) aimm_->setDt(dt); }
        else { if (ukf_) ukf_->setDt(dt); }
    }

    void Tracker::filterPredict() 
    {
        if (use_aimm_) { if (aimm_) aimm_->predict(); }
        else { if (ukf_) ukf_->predict(); }
    }

    void Tracker::filterUpdate(const Eigen::VectorXd& z_meas) 
    {
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

    void Tracker::filterSetMeasurementNoise(const Eigen::MatrixXd& R) 
    {
        if (use_aimm_) { if (aimm_) aimm_->setMeasurementNoise(R); }
        else { if (ukf_) ukf_->setMeasurementNoise(R); }
    }

    void Tracker::handleDetectingState(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt)
    {
        (void)z_init;
        (void)dt;
        if (has_data) {
            detect_fail_count_ = 0;

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
                if(state_log_mode_)
                {
                    ROS_WARN("[Tracker] STATE: DETECTING -> TRACKING");
                }
            }
        } 
        else 
        {   
            detect_fail_count_++;
            if (detect_fail_count_ > max_lost_count_) {
                track_state_ = rm_track::LOST;
                detect_fail_count_ = 0;
                q_track_valid_ = false;
                if(state_log_mode_)
                {
                    ROS_WARN("[Tracker] STATE: DETECTING -> LOST");
                }
            }
        }
    }

    void Tracker::handleTrackingState(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt)
    {
        (void)z_init;
        (void)dt;
        if (has_data) {
            lost_count_ = 0;
            if (is_lidar_msg && !is_cam_msg) 
            {
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
            if (tracking_with_cam_) 
            {
                cam_lost_counter_ = 1;
                lost_count_ = cam_lost_counter_;
            } else {
                lidar_lost_counter_ = 1;
                lost_count_ = lidar_lost_counter_;
            }
            is_tracking_ = false;
            if(state_log_mode_)
            {
                ROS_WARN("[Tracker] STATE: TRACKING -> TEMP_LOST");
            }
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
            if (is_lidar_msg && !is_cam_msg) 
            {
                tracking_with_cam_ = false;
                lidar_lost_counter_ = 0;
            } else {
                tracking_with_cam_ = true;
                cam_lost_counter_ = 0;
            }
            lost_count_ = 0;
            filterUpdate(z_meas);
            if(state_log_mode_)
            {
                ROS_WARN("[Tracker] STATE: TEMP_LOST -> TRACKING");
            }
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
                if(state_log_mode_)
                {
                    ROS_WARN("[Tracker] STATE: TEMP_LOST -> LOST");
                }
                
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
                if(state_log_mode_)
                {
                    ROS_WARN("[Tracker] STATE: LOST -> DETECTING (AIMM)");
                }
            } else if (!use_aimm_ && ukf_) {
                if (ukf_ && ukf_->getModel()) {
                    ukf_->initialize(z_init);
                    if(state_log_mode_)
                    {
                        ROS_WARN("[Tracker] STATE: LOST -> DETECTING (UKF)");
                    }
                } else {
                    if(state_log_mode_)
                    {
                        ROS_ERROR("[Tracker] Cannot initialize UKF: ukf or model is null");
                    }
                }
            }
        }
    }

    void Tracker::processTracking(bool has_data, bool is_cam_msg, bool is_lidar_msg, const Eigen::VectorXd& z_meas, const Eigen::Vector3d& z_init, double dt)
    {
        if(track_state_ != rm_track::LOST && (dt < 0.0001 || dt > 1.0)) return;

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

        rm_msgs::GimbalCmd track_data_msg;
        track_data_msg.mode = 2;
        track_data_msg.target_pos.header.stamp = current_time_;
        track_data_msg.target_pos.header.frame_id = target_frame_;

        rm_msgs::TrackData img_track_data_msg;
        img_track_data_msg.header.stamp = current_time_;
        img_track_data_msg.header.frame_id = target_frame_;

        img_track_data_msg.rotation.w = q_track_.getW();
        img_track_data_msg.rotation.x = q_track_.getX();
        img_track_data_msg.rotation.y = q_track_.getY();
        img_track_data_msg.rotation.z = q_track_.getZ();
        img_track_data_msg.tracking = is_tracking_;

        if(filterIsInitialized()) {
            Eigen::VectorXd state = filterGetState();

            if(state.size() >= 6) {
                track_data_msg.target_pos.point.x = state(0);
                track_data_msg.target_pos.point.y = state(1);
                track_data_msg.target_pos.point.z = state(2);

                img_track_data_msg.position.x = state(0);
                img_track_data_msg.position.y = state(1);
                img_track_data_msg.position.z = state(2);
                img_track_data_msg.velocity.x = state(3);
                img_track_data_msg.velocity.y = state(4);
                img_track_data_msg.velocity.z = state(5);
                
                last_known_position_ << state(0), state(1), state(2);
                last_known_velocity_ << state(3), state(4), state(5);
            }
        } else {
            track_data_msg.target_pos.point.x = last_known_position_(0);
            track_data_msg.target_pos.point.y = last_known_position_(1);
            track_data_msg.target_pos.point.z = last_known_position_(2);

            img_track_data_msg.position.x = last_known_position_(0);
            img_track_data_msg.position.y = last_known_position_(1);
            img_track_data_msg.position.z = last_known_position_(2);
            img_track_data_msg.velocity.x = last_known_velocity_(0);
            img_track_data_msg.velocity.y = last_known_velocity_(1);
            img_track_data_msg.velocity.z = last_known_velocity_(2);
        }
        
        tracker_pub_.publish(track_data_msg);
        img_tracker_pub_.publish(img_track_data_msg);
        if (q_track_valid_) 
        {
            geometry_msgs::TransformStamped track_tf;
            track_tf.header.stamp = current_time_;
            track_tf.header.frame_id = target_frame_;
            track_tf.child_frame_id = "tracked_target_" ;
            track_tf.transform.translation.x = track_data_msg.target_pos.point.x;
            track_tf.transform.translation.y = track_data_msg.target_pos.point.y;
            track_tf.transform.translation.z = track_data_msg.target_pos.point.z;
            track_tf.transform.rotation.w = q_track_.getW();
            track_tf.transform.rotation.x = q_track_.getX();
            track_tf.transform.rotation.y = q_track_.getY();
            track_tf.transform.rotation.z = q_track_.getZ();
            tf_broadcaster_->sendTransform(track_tf);   
        }
    }
}