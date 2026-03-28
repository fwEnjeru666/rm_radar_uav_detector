#include <rm_track.h>

PLUGINLIB_EXPORT_CLASS(rm_radarplugin::Tracker, nodelet::Nodelet);

namespace rm_radarplugin
{
    void Tracker::onInit()
    {
        ros::NodeHandle nh = getMTPrivateNodeHandle();
        
        initialize(nh);

        //pose solver init
        pose_solver_.onInit(nh);
        
    }

    void Tracker::initialize(ros::NodeHandle &nh)
    {
        nh_ = ros::NodeHandle(nh, "radar_track");
        ROS_INFO("radar track initialized");

        // Read AIMM switch from parameter server (can also be toggled via dynamic_reconfigure)
        use_aimm_ = nh_.param("use_aimm", false);

        ukf_ = rm_radarplugin::UKF(nh_);
        ukf_.initDynamicReconfigure();  // 初始化 dynamic_reconfigure server

        if (use_aimm_)
        {
            aimm_ = rm_radarplugin::AIMM(nh_);
            aimm_.initDynamicReconfigure();
            ROS_INFO("[Tracker] Using AIMM (Adaptive Interacting Multiple Model) estimator");
        }
        else
        {
            ROS_INFO("[Tracker] Using single UKF estimator");
        }
        
        // 订阅 PoseSolver 输出的带有3D位姿的检测消息
        detection_sub_ = nh_.subscribe("/pose_solver/solved_pose", 10, &Tracker::camDetectionCB, this);
        
        // 订阅 LiDAR 检测消息
        lidar_detection_sub_ = nh_.subscribe("/lidar_detector/lidar_detection", 1, &Tracker::lidarDetectionCB, this);
        
        target_frame_ = nh_.param("target_frame", std::string("odom"));
        hit_threshold_ = nh_.param("hit_threshold", 3);
        max_lost_count_ = nh_.param("max_lost_count", 5);
        debug_mode_ = nh_.param("debug_mode", false);
        use_lidar_fusion_ = nh_.param("use_lidar_fusion", true);
        lidar_timeout_ = nh_.param("lidar_timeout", 0.5);
        vision_timeout_ = nh_.param("vision_timeout", 0.3);
        
        tracker_pub_ = nh_.advertise<rm_msgs::TrackData>("/tracker/track_data", 1);
        aimm_debug_pub_ = nh_.advertise<rm_radar_msgs::aimm_debugger>("/tracker/aimm_debug", 1);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(ros::Duration(10));
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ =  std::make_shared<tf2_ros::TransformBroadcaster>();

        //dynamic reconfigure
        track_cfg_srv_ = new dynamic_reconfigure::Server<rm_track::trackConfig>(nh_);
        track_cfg_cb_ = boost::bind(&Tracker::trackconfigCB, this, _1, _2);
        track_cfg_srv_->setCallback(track_cfg_cb_);


        //timer
        detection_timer_ = nh_.createTimer(ros::Duration(0.01), &Tracker::detectionCB, this);  

    }

    void Tracker::trackconfigCB(rm_track::trackConfig& config, uint32_t level)
    {
        hit_threshold_ = config.hit_threshold;
        max_lost_count_ = config.max_lost_count;
        debug_mode_ = config.debug_mode;
        aimm_debug_mode_ = config.aimm_debug_mode;

        // Handle AIMM toggle at runtime
        bool new_use_aimm = config.use_aimm;
        if (new_use_aimm != use_aimm_)
        {
            use_aimm_ = new_use_aimm;
            if (use_aimm_ && !aimm_.isInitialized())
            {
                aimm_ = rm_radarplugin::AIMM(nh_);
                aimm_.initDynamicReconfigure();
            }
            // Force re-detection so the new estimator gets initialized cleanly
            track_state_ = rm_track::LOST;
            is_tracking_ = false;
            q_track_valid_ = false;
            ROS_INFO("[Tracker] Switched estimator to %s — resetting to LOST state",
                     use_aimm_ ? "AIMM" : "UKF");
        }

        ROS_INFO("Reconfigure Request: hit_threshold=%d, max_lost_count=%d, debug_mode=%d, use_aimm=%d",
                 hit_threshold_, max_lost_count_, debug_mode_, use_aimm_);
    }

    void Tracker::lidarDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection)
    {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        latest_lidar_detection_ = detection;
        // Use message stamp (works with sim time / rosbag)
        last_lidar_time_ = detection->header.stamp;
        
        if (debug_mode_) {
            ROS_INFO_THROTTLE(1, "[Tracker] LiDAR detection: (%.3f, %.3f, %.3f)",
                detection->pose.position.x, detection->pose.position.y, detection->pose.position.z);
        }
    }

    void Tracker::camDetectionCB(const rm_radar_msgs::DroneDetection::ConstPtr& detection)
    {
        std::lock_guard<std::mutex> lock(cam_mutex_);
        latest_cam_detection_ = detection;
        // Use message stamp (works with sim time / rosbag)
        last_cam_time_ = detection->header.stamp;

        if (debug_mode_)
        {       
            ROS_INFO_THROTTLE(1, "[Tracker] Vision detection: (%.3f, %.3f, %.3f)",
                detection->pose.position.x, detection->pose.position.y, detection->pose.position.z);
        }
        
    }

    bool Tracker::transform2Odom(const std_msgs::Header& header, const Eigen::Vector3d& p_cam, Eigen::Vector3d& p_odom)
    {
        //1. warp up data
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
                target_frame_, header.frame_id,
                header.stamp,          
                ros::Duration(0.1));
            tf2::doTransform(point_in, point_out, transform_stamped);
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN_THROTTLE(2.0, "Transform failure [%s -> %s]: %s",
                header.frame_id.c_str(), target_frame_.c_str(), ex.what());
            try
            {
                geometry_msgs::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                    target_frame_, header.frame_id,
                    ros::Time(0),       
                    ros::Duration(0.05));
                tf2::doTransform(point_in, point_out, transform_stamped);
            }
            catch(tf2::TransformException &ex2)
            {
                ROS_WARN_THROTTLE(2.0, "Fallback transform failure [%s -> %s]: %s",
                    header.frame_id.c_str(), target_frame_.c_str(), ex2.what());
                return false;
            }
        }
        p_odom = Eigen::Vector3d(point_out.point.x, point_out.point.y, point_out.point.z);
        return true;

    }

    void Tracker::detectionCB(const ros::TimerEvent& event)
    {   
        // Use a time base consistent with incoming messages (important for /use_sim_time / rosbag)
        ros::Time now = ros::Time::now();

        rm_radar_msgs::DroneDetection::ConstPtr current_lidar_msg_ = nullptr;
        rm_radar_msgs::DroneDetection::ConstPtr current_cam_msg_ = nullptr;

        // 1) Grab latest messages (if any)
        {
            std::lock_guard<std::mutex> lock(cam_mutex_);
            if (latest_cam_detection_) {
                current_cam_msg_ = latest_cam_detection_;
                latest_cam_detection_ = nullptr;
            }
        }
        {
            std::lock_guard<std::mutex> lock(lidar_mutex_);
            if (latest_lidar_detection_) {
                current_lidar_msg_ = latest_lidar_detection_;
                latest_lidar_detection_ = nullptr;
            }
        }

        // 2) Pick a time reference and apply timeouts based on message stamps, not wall time
        // Prefer vision stamp if present, else lidar stamp, else wall time.
        if (current_cam_msg_ && !current_cam_msg_->header.stamp.isZero()) now = current_cam_msg_->header.stamp;
        else if (current_lidar_msg_ && !current_lidar_msg_->header.stamp.isZero()) now = current_lidar_msg_->header.stamp;

        // init time base
        if (last_time_.isZero()) {
            last_time_ = now;
            return;
        }

        double dt = (now - last_time_).toSec();
        if (dt < 0.001 || dt > 1.0) {
            last_time_ = now;
            return;
        }

        Eigen::Vector3d z_meas = Eigen::Vector3d::Zero();
        bool has_data = false;
        bool is_vision_active = false;

        // 3) Vision first (apply timeout using header stamps)
        if (current_cam_msg_ && !last_cam_time_.isZero() &&
            (now - last_cam_time_).toSec() < vision_timeout_)
        {
            Eigen::Vector3d p_cam(current_cam_msg_->pose.position.x,
                                  current_cam_msg_->pose.position.y,
                                  current_cam_msg_->pose.position.z);
            Eigen::Vector3d p_odom;

            if (p_cam.norm() >= 0.01 && transform2Odom(current_cam_msg_->header, p_cam, p_odom))
            {
                z_meas = p_odom;
                has_data = true;
                is_vision_active = true;
                current_time_ = current_cam_msg_->header.stamp;

                ROS_INFO_THROTTLE(1, "[Tracker] Vision OK: cam(%.2f,%.2f,%.2f) -> odom(%.2f,%.2f,%.2f) state=%d",
                    p_cam.x(), p_cam.y(), p_cam.z(),
                    p_odom.x(), p_odom.y(), p_odom.z(), track_state_);

                // Update quaternion only when using vision
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

        // 4) LiDAR fallback (apply timeout using header stamps)
        if (!has_data && use_lidar_fusion_ && current_lidar_msg_ && !last_lidar_time_.isZero() &&
            (now - last_lidar_time_).toSec() < lidar_timeout_)
        {
            Eigen::Vector3d p_lidar(current_lidar_msg_->pose.position.x,
                                    current_lidar_msg_->pose.position.y,
                                    current_lidar_msg_->pose.position.z);
            Eigen::Vector3d p_odom;

            if (transform2Odom(current_lidar_msg_->header, p_lidar, p_odom))
            {
                z_meas = p_odom;
                has_data = true;
                current_time_ = current_lidar_msg_->header.stamp;
                ROS_WARN_THROTTLE(1.0, "[Tracker] Vision lost! Using LiDAR fallback.");
            }
        }

        // 5) Adjust measurement noise / run state machine (unchanged)
        if (has_data && filterIsInitialized()) 
        {
            Eigen::MatrixXd R_base = filterGetBaseMeasurementNoise();
            Eigen::Matrix3d R_matrix = Eigen::Matrix3d::Zero();
            
            if (is_vision_active) 
            {
                // 视觉模式: 信任XY，不信任Z
                R_matrix(0,0) = R_base(0,0) * 1.0;   // XY 缩小噪声 → 更信任
                R_matrix(1,1) = R_base(1,1) * 1.0;
                R_matrix(2,2) = R_base(2,2) * 10.0;   // Z 放大噪声 → 不信任
            } 
            else 
            {
                // LiDAR模式: 信任Z，不信任XY
                R_matrix(0,0) = R_base(0,0) * 5.0;   // XY 放大噪声 → 不信任
                R_matrix(1,1) = R_base(1,1) * 5.0;
                R_matrix(2,2) = R_base(2,2) * 0.1;   // Z 缩小噪声 → 更信任
            }
            
            filterSetMeasurementNoise(R_matrix);
            
            ROS_DEBUG_THROTTLE(1, "[Tracker] R adjusted: diag(%.4f, %.4f, %.4f) src=%s",
                R_matrix(0,0), R_matrix(1,1), R_matrix(2,2),
                is_vision_active ? "VISION" : "LIDAR");
        }

        // 5. 驱动状态机
        if (!has_data) 
        {
            current_time_ = now; // 没有数据时，以当前时间进行推演
        }
        processTracking(has_data, z_meas, dt);
        last_time_ = now;
    }

    void Tracker::init_ukf(const Eigen::Vector3d& z_meas)
    {
        // 根据当前选择的模型类型初始化正确维度的状态向量
        int model_type = ukf_.getModelType();
        int state_dim = rm_radarplugin::UKF::getStateDimForModel(model_type);
        
        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(state_dim);
        Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(state_dim, state_dim);
        
        // 位置初始化为测量值
        x0.head(3) = z_meas;
        
        switch(model_type)
        {
            case rm_track::CV: // [x, y, z, vx, vy, vz]
            {
                x0.tail(3).setZero();  // 速度初始化为0
                P0.topLeftCorner(3,3) *= 0.1;      // 位置协方差较小
                P0.bottomRightCorner(3,3) *= 10.0; // 速度协方差较大
                ROS_INFO("Init UKF with CV model (6D state)");
                break;
            }
            case rm_track::CA: // [x, y, z, vx, vy, vz, ax, ay, az]
            {
                x0.segment(3, 3).setZero();  // 速度初始化为0
                x0.segment(6, 3).setZero();  // 加速度初始化为0
                P0.block<3,3>(0,0) *= 0.1;   // 位置协方差
                P0.block<3,3>(3,3) *= 10.0;  // 速度协方差
                P0.block<3,3>(6,6) *= 1.0;   // 加速度协方差
                ROS_INFO("Init UKF with CA model (9D state)");
                break;
            }
            case rm_track::CTRV: // [x, y, z, v, yaw, yaw_rate]
            {
                x0(3) = 0.0;  // 速度初始化为0
                x0(4) = 0.0;  // yaw 初始化为0
                x0(5) = 0.0;  // yaw_rate 初始化为0
                P0.topLeftCorner(3,3) *= 0.1;  // 位置协方差
                P0(3,3) = 10.0;   // v 协方差
                P0(4,4) = 0.5;    // yaw 协方差
                P0(5,5) = 0.1;    // yaw_rate 协方差
                ROS_INFO("Init UKF with CTRV model (6D state)");
                break;
            }
            default:
            {
                // 默认使用CV模型
                x0.tail(3).setZero();
                P0.topLeftCorner(3,3) *= 0.1;
                P0.bottomRightCorner(3,3) *= 10.0;
                ROS_WARN("Unknown model type %d, using CV model", model_type);
                break;
            }
        }
        
        ukf_.initialize(x0, P0);
    }

    // ============================================================
    // init_aimm: initialize the AIMM estimator with first measurement
    // ============================================================
    void Tracker::init_aimm(const Eigen::Vector3d& z_meas)
    {
        aimm_.initialize(z_meas);
        ROS_INFO("[Tracker] AIMM initialized at (%.3f, %.3f, %.3f)", z_meas.x(), z_meas.y(), z_meas.z());
    }

    // ============================================================
    // Unified filter interface — dispatches to UKF or AIMM
    // ============================================================
    bool Tracker::filterIsInitialized() const
    {
        return use_aimm_ ? aimm_.isInitialized() : ukf_.isInitialized();
    }

    void Tracker::filterSetDt(double dt)
    {
        if (use_aimm_)
            aimm_.setDt(dt);
        else
            ukf_.setDt(dt);
    }

    void Tracker::filterPredict()
    {
        if (use_aimm_)
            aimm_.predict();
        else
            ukf_.predict();
    }

    void Tracker::filterUpdate(const Eigen::Vector3d& z_meas)
    {
        if (use_aimm_)
            aimm_.update(z_meas);
        else
            ukf_.update(z_meas);
    }

    Eigen::VectorXd Tracker::filterGetState() const
    {
        return use_aimm_ ? aimm_.getState() : ukf_.getState();
    }

    Eigen::MatrixXd Tracker::filterGetBaseMeasurementNoise() const
    {
        return use_aimm_ ? aimm_.getBaseMeasurementNoise() : ukf_.getBaseMeasurementNoise();
    }

    void Tracker::filterSetMeasurementNoise(const Eigen::MatrixXd& R)
    {
        if (use_aimm_)
            aimm_.setMeasurementNoise(R);
        else
            ukf_.setMeasurementNoise(R);
    }

//state handling......................................
    void Tracker::handleDetectingState(bool has_data, const Eigen::Vector3d& z_meas, double dt)
    {
        if (has_data)
        {
            detect_fail_count_ = 0;  
            detected_count_ ++;
            filterSetDt(dt);
            filterPredict();
            filterUpdate(z_meas);
            ROS_INFO("[Tracker] STATE: DETECTING, count=%d/%d", detected_count_, hit_threshold_);
            // Enter TRACKING as soon as we have enough consecutive hits.
            // Using '>=' avoids requiring an extra frame (off-by-one) which makes tracking feel slow.
            if(detected_count_ >= hit_threshold_)
         {
             track_state_ = rm_track::TRACKING;
             is_tracking_ = true; 
             ROS_INFO("[Tracker] STATE: DETECTING -> TRACKING! (detected %d times)", detected_count_);
             detected_count_ = 0; //reset counter
         }
        }
        else
        {
            // 允许一定次数的丢帧
            detect_fail_count_++;
            ROS_INFO("[Tracker] STATE: DETECTING, no data this frame (fail %d/%d)", detect_fail_count_, max_lost_count_);
            if (detect_fail_count_ > max_lost_count_) {
                ROS_INFO("[Tracker] STATE: DETECTING -> LOST (too many consecutive failures)");
                track_state_ = rm_track::LOST;
                detected_count_ = 0;
                detect_fail_count_ = 0;
                q_track_valid_ = false;
            }
        }
    }

    void Tracker::handleTrackingState(bool has_data, const Eigen::Vector3d& z_meas, double dt)
    {
        if (has_data)
        {
            lost_count_ = 0;
            filterSetDt(dt);
            filterPredict();
            filterUpdate(z_meas);
            is_tracking_ = true;
            if(debug_mode_) ROS_INFO("STATE:%d: Tracking with detection.", track_state_);
        }
        else
        {
            if(debug_mode_) ROS_INFO("STATE:%d: Track state changed to TEMP_LOST--no detection.", track_state_);
            track_state_ = rm_track::TEMP_LOST;
            lost_count_ = 1;
        }
    }

    void Tracker::handleTempLostState(bool has_data, const Eigen::Vector3d& z_meas, double dt)
    {
        if (has_data)
        {
            track_state_ = rm_track::TRACKING;
            is_tracking_ = true; 
            filterSetDt(dt);
            filterPredict();
            filterUpdate(z_meas);
            if(debug_mode_) ROS_INFO("STATE:%d: Track state changed to TRACKING--detection reacquired.", track_state_);
        }
        else
        {
            lost_count_ ++;
            if(lost_count_ > max_lost_count_)
            {
                track_state_ = rm_track::LOST;
                is_tracking_ = false;
                q_track_valid_ = false; 
                if(debug_mode_) ROS_INFO("STATE:%d: Track state changed to LOST--too many consecutive lost.", track_state_);
            }
            else
            {
                filterSetDt(dt);
                filterPredict();
                if(debug_mode_) ROS_INFO("STATE:%d: TEMP_LOST without detection.", track_state_);
            }
        }
    }

    void Tracker::handleLostState(bool has_data, const Eigen::Vector3d& z_meas, double dt)
    {
        if (has_data)
        {
            track_state_ = rm_track::DETECTING;
            detected_count_ = 1;
            detect_fail_count_ = 0;

            if (use_aimm_)
                init_aimm(z_meas);
            else
                init_ukf(z_meas);

            ROS_INFO("[Tracker] STATE: LOST -> DETECTING (%s), first detection at (%.3f, %.3f, %.3f)", 
                use_aimm_ ? "AIMM" : "UKF",
                z_meas.x(), z_meas.y(), z_meas.z());
        }
        else
        {
            ROS_INFO_THROTTLE(3, "[Tracker] STATE: LOST, waiting for valid detection... "
                              "(cam_sub=%d, lidar_sub=%d, cam_fresh=%d, lidar_fresh=%d)",
                              (detection_sub_.getNumPublishers() > 0),
                              (lidar_detection_sub_.getNumPublishers() > 0),
                              (!last_cam_time_.isZero() && (ros::Time::now() - last_cam_time_).toSec() < vision_timeout_),
                              (!last_lidar_time_.isZero() && (ros::Time::now() - last_lidar_time_).toSec() < lidar_timeout_));
        }
    }


    void Tracker::processTracking(bool has_data, const Eigen::Vector3d& z_meas, double dt)
    {
        if(track_state_ != rm_track::LOST)
        {
            if(dt < 0.0001 || dt > 1.0)
            {
                ROS_WARN_ONCE("Unreasonable dt: %f, skip this frame.", dt);
                return;
            }
        }

        switch (track_state_)
        {
            case rm_track::DETECTING:
                handleDetectingState(has_data, z_meas, dt);
                break;
            case rm_track::TRACKING:
                handleTrackingState(has_data, z_meas, dt);
                break;
            case rm_track::TEMP_LOST:
                handleTempLostState(has_data, z_meas, dt);
                break;
            case rm_track::LOST:
                handleLostState(has_data, z_meas, dt);
                break;
            default:
                ROS_WARN("Unknown track state!");
                break;
        }

        //publish track data
        if(track_state_ == rm_track::TRACKING || track_state_ == rm_track::TEMP_LOST)
        {
            publishTrackerData();
        }
    }

    void Tracker::publishTrackerData()
    {
        // publish AIMM debug data if enabled
        if (use_aimm_ && aimm_debug_mode_)
        {
            rm_radar_msgs::aimm_debugger aimm_msg = aimm_.getDebugMsg();
            aimm_msg.header.stamp = ros::Time::now();
            aimm_msg.header.frame_id = target_frame_;
            
            // Sync internal UKF messages headers
            aimm_msg.cv_debug.header = aimm_msg.header;
            aimm_msg.ca_debug.header = aimm_msg.header;
            aimm_msg.ctrv_debug.header = aimm_msg.header;

            aimm_debug_pub_.publish(aimm_msg);
        }

        //publish track data
        rm_msgs::TrackData track_data_msg;
        track_data_msg.header.stamp = current_time_;
        track_data_msg.header.frame_id = target_frame_;

        track_data_msg.rotation.w = q_track_.getW();
        track_data_msg.rotation.x = q_track_.getX();
        track_data_msg.rotation.y = q_track_.getY();
        track_data_msg.rotation.z = q_track_.getZ();

        if(!filterIsInitialized())
         {
             ROS_WARN("Filter not initialized, cannot publish track data.");
             return;
         }
        Eigen::VectorXd state = filterGetState();

        track_data_msg.id = track_id_;
        track_data_msg.tracking = is_tracking_;

        if(state.size()>=6)
        {
            track_data_msg.position.x = state(0);
            track_data_msg.position.y = state(1);
            track_data_msg.position.z = state(2);
            track_data_msg.velocity.x = state(3);
            track_data_msg.velocity.y = state(4);
            track_data_msg.velocity.z = state(5);
        }
        tracker_pub_.publish(track_data_msg);

        // Log AIMM model info if active
        if (use_aimm_ && aimm_.isInitialized() && aimm_debug_mode_)
        {
            Eigen::VectorXd probs = aimm_.getModelProbabilities();
            ROS_INFO_THROTTLE(1, "[Tracker] AIMM probs: CV=%.1f%% CA=%.1f%% CTRV=%.1f%% | Active: %s | λ=%.2f",
                              probs(0)*100, probs(1)*100, probs(2)*100,
                              aimm_.getActiveModelName().c_str(),
                              aimm_.getManeuverIndicator());
        }

        //broadcast tf  
        if (q_track_valid_) {
            geometry_msgs::TransformStamped track_tf;
            // NOTE:
            // - TrackData should keep `current_time_` (measurement time) for downstream fusion.
            // - TF should use `ros::Time::now()` to avoid TF_OLD_DATA / time-jump issues when
            //   bag timestamps jump or upstream stamps are not monotonic.
            ros::Time tf_stamp = ros::Time::now();
            if (tf_stamp.isZero()) {
                tf_stamp = current_time_;
            }

            track_tf.header.stamp = tf_stamp;
            track_tf.header.frame_id = target_frame_;
            track_tf.child_frame_id = "tracked_target_" + std::to_string(track_id_);
            track_tf.transform.translation.x = state(0);
            track_tf.transform.translation.y = state(1);
            track_tf.transform.translation.z = state(2);
            track_tf.transform.rotation.w = q_track_.getW();
            track_tf.transform.rotation.x = q_track_.getX();
            track_tf.transform.rotation.y = q_track_.getY();
            track_tf.transform.rotation.z = q_track_.getZ();
            tf_broadcaster_->sendTransform(track_tf);
        }
    }
    
    // TODO : debug msg for TRACKER
}
