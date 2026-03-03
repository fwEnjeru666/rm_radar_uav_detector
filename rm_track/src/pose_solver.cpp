#include "pose_solver.h"

namespace rm_radarplugin
{
    void PoseSolver::onInit(ros::NodeHandle &nh)
    {
        initialize(nh);
        ROS_INFO("pose solver onInit finished");
    }

    void PoseSolver::initialize(ros::NodeHandle &nh)
    {
        nh_ = ros::NodeHandle(nh, "pose_solver");
        ROS_INFO("pose solver initialized");

        // cam info
        if (!nh_.getParam("cam_info_topic", cam_info_topic_))
        {
            cam_info_topic_ = "/hk_camera/camera_info";
            ROS_WARN("Parameter 'cam_info_topic' not found. Using default: %s", cam_info_topic_.c_str());
        }
        cam_info_sub_ = nh_.subscribe(cam_info_topic_, 1, &PoseSolver::camInfoCB, this); 
        if(!intrinsics_.empty() && !dist_coeffs_.empty())
        {
            getReprojectParams();
        }

        //track data sub
        track_data_sub_ = nh_.subscribe("/processor/result_msg", 1, &PoseSolver::PoseSolverCB, this);
        
        //solved pose pub (发布给Tracker使用)
        solved_pose_pub_ = nh_.advertise<rm_radar_msgs::DroneDetection>("/pose_solver/solved_pose", 1);

        //tf
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>();
    }

    bool PoseSolver::solvePose()
    {
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(armor_3d_points_, armor_2d_points_, intrinsics_, dist_coeffs_, rvec, tvec);
        if (!success)
        {
            ROS_ERROR("Could not solve PnP");
            return false;
        }
        double tx = tvec.at<double>(0);
        double ty = tvec.at<double>(1);
        double tz = tvec.at<double>(2);

        // 过滤明显异常的 PnP 结果
        // z 必须为正且在合理范围内 (0.3m ~ 6m)
        if (tz < 0.3 || tz > 6.0 ||
            std::abs(tx) > tz * 2.0 ||  // x 偏移不能超过距离的 2 倍
            std::abs(ty) > tz * 2.0 ||  // y 偏移不能超过距离的 2 倍
            std::isnan(tx) || std::isnan(ty) || std::isnan(tz))
        {
            return false;  
        } 

        Eigen::Vector3d rvec_eigen;
        rvec_eigen << rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2);
        double angle = rvec_eigen.norm();
        if(angle < 1e-6)
            q_ = Eigen::Quaterniond::Identity();
        else
        {
            Eigen::Vector3d axis = rvec_eigen / angle;
            q_ = Eigen::AngleAxisd(angle, axis);
        }
        tvec_ = Eigen::Vector3d(tx, ty, tz);
            
        return true;

    }

    double PoseSolver::calReprojectCost(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                                        const cv::Mat& K, const cv::Mat& D,
                                        double current_yaw, double yaw_center)
    {
        const double r00 = R(0,0), r01 = R(0,1), r02 = R(0,2);
        const double r10 = R(1,0), r11 = R(1,1), r12 = R(1,2);
        const double r20 = R(2,0), r21 = R(2,1), r22 = R(2,2);
        const double tx = t(0), ty = t(1), tz = t(2);

        double total_cost = 0.0;
        const size_t n = armor_3d_points_.size();
        
        std::vector<double> u_arr(n), v_arr(n);

        // Loop 1: 投影 + 位置误差
        for (size_t i = 0; i < n; i++)
        {
            double X = armor_3d_points_[i].x;
            double Y = armor_3d_points_[i].y;
            double Z = armor_3d_points_[i].z;

            double xc = r00 * X + r01 * Y + r02 * Z + tx;
            double yc = r10 * X + r11 * Y + r12 * Z + ty;
            double zc = r20 * X + r21 * Y + r22 * Z + tz;

            if (zc < 1e-5) zc = 1e-5;
            double inv_z = 1.0 / zc;
            double x = xc * inv_z;
            double y = yc * inv_z;

            double r2 = x * x + y * y;
            double r4 = r2 * r2;
            double c_dist = 1.0 + k1_ * r2 + k2_ * r4 + k3_ * r4 * r2;
            double xy2 = 2.0 * x * y;
            double x_dist = x * c_dist + p1_ * xy2 + p2_ * (r2 + 2.0 * x * x);
            double y_dist = y * c_dist + p1_ * (r2 + 2.0 * y * y) + p2_ * xy2;

            double u_proj = fx_ * x_dist + cx_;
            double v_proj = fy_ * y_dist + cy_;
            
            u_arr[i] = u_proj;
            v_arr[i] = v_proj;

            double dx = armor_2d_points_[i].x - u_proj;
            double dy = armor_2d_points_[i].y - v_proj;

            double cost = std::sqrt(dx * dx + dy * dy);
            total_cost += huberLoss(cost, opt_params_.huber_delta_px);
        }

        // Loop 2: 形状误差
        for(size_t i = 0; i < n; i++)
        {
            size_t next_i = (i + 1) % n;
            double v_proj_x = u_arr[next_i] - u_arr[i];
            double v_proj_y = v_arr[next_i] - v_arr[i];

            double v_obs_x = armor_2d_points_[next_i].x - armor_2d_points_[i].x;
            double v_obs_y = armor_2d_points_[next_i].y - armor_2d_points_[i].y;

            double angle = getAngle(v_proj_x, v_proj_y, v_obs_x, v_obs_y);
            double angle_deg = angle * 180.0 / CV_PI;
            total_cost += huberLoss(angle_deg, opt_params_.huber_delta_deg);
        }
        
        if (opt_params_.prior_weight > 0.0) 
        {
        double yaw_diff = std::abs(current_yaw - yaw_center);
        // 对偏离进行二次惩罚
        total_cost += opt_params_.prior_weight * yaw_diff * yaw_diff;
        }

        return total_cost;
    }

    void PoseSolver::optimizeTranslation(const Eigen::Matrix3d& R)
    {
        // Safety check: ensure undistorted_points_ has 4 elements
        if (undistorted_points_.size() != 4) {
            ROS_WARN_THROTTLE(1, "undistorted_points_ size is %lu, expected 4. Skipping translation optimization.", 
                undistorted_points_.size());
            return;
        }
        
        // Implementation of translation optimization
        // armor_3d_points_ 有 4 个点，所以是 8 行
        Eigen::Matrix<double, 8, 3> A;
        Eigen::Matrix<double, 8, 1> b;
        
        const double r00 = R(0,0), r01 = R(0,1), r02 = R(0,2);
        const double r10 = R(1,0), r11 = R(1,1), r12 = R(1,2);
        const double r20 = R(2,0), r21 = R(2,1), r22 = R(2,2);

        for(size_t i = 0; i < 4; i++)
        {   
            double mx = undistorted_points_[i].x;
            double my = undistorted_points_[i].y;

            double X = armor_3d_points_[i].x;
            double Y = armor_3d_points_[i].y;
            double Z = armor_3d_points_[i].z;

            double RP_x = r00 * X + r01 * Y + r02 * Z;
            double RP_y = r10 * X + r11 * Y + r12 * Z;
            double RP_z = r20 * X + r21 * Y + r22 * Z;

            size_t row = 2*i;
            A(row, 0) = 1.0; A(row, 1) = 0.0; A(row, 2) = -mx;
            b(row, 0) = mx * RP_z - RP_x;

            A(row+1, 0) = 0.0; A(row+1, 1) = 1.0; A(row+1, 2) = -my;
            b(row+1, 0) = my * RP_z - RP_y;
        }
        
        //(A^T * A) * x = A^T * b
        tvec_ = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    }

    void PoseSolver::optimizeYaw()
    {
        // Implementation of yaw optimization
        const double pitch = opt_params_.fixed_pitch;
        Eigen::Quaterniond q_init(q_.w(), q_.x(), q_.y(), q_.z());
        Eigen::Matrix3d R_init = q_init.toRotationMatrix();
        
        double yaw_center = std::atan2(R_init(0, 2), R_init(2, 2));

        double best_yaw = yaw_center;
        Eigen::Vector3d best_t = tvec_;
        double min_error = std::numeric_limits<double>::max();

        auto getRotationMatrix = [&](double yaw) -> Eigen::Matrix3d {
            double s_y = std::sin(yaw), c_y = std::cos(yaw);
            double s_p = std::sin(pitch), c_p = std::cos(pitch);
            Eigen::Matrix3d R;
            R << c_y, s_y * s_p, s_y * c_p,
                0.0, c_p,       -s_p,
                -s_y, c_y * s_p, c_y * c_p;
            return R;
        };

        auto runSearch = [&](double start_yaw, double range, double step) {
            for (double delta = -range; delta <= range; delta += step)
            {
                double current_yaw = start_yaw + delta * (CV_PI / 180.0);
                Eigen::Matrix3d R = getRotationMatrix(current_yaw);

                Eigen::Vector3d t_opt = tvec_;
                if (opt_params_.optimize_translation) {
                    optimizeTranslation(R);
                    t_opt = tvec_;
                }

                double cost = calReprojectCost(R, t_opt, intrinsics_, dist_coeffs_, current_yaw, yaw_center);

                if (cost < min_error) {
                    min_error = cost;
                    best_yaw = current_yaw;
                    best_t = t_opt;
                }
            }
        };

        // First run a coarse search around the initial yaw
        runSearch(yaw_center, opt_params_.big_range, opt_params_.big_step);
        // Then run a finer search around the best yaw found
        runSearch(best_yaw, opt_params_.small_range, opt_params_.small_step);

        // Update the optimized pose
        Eigen::Matrix3d R_final = getRotationMatrix(best_yaw);
        optimized_q_ = Eigen::Quaterniond(R_final);
        optimized_q_.normalize();
        tvec_ = best_t;
    }



    // pose solver callback - handles DroneDetectionArray
    void PoseSolver::PoseSolverCB(const rm_radar_msgs::DroneDetectionArray::ConstPtr& detections)
    {
        if (detections->detections.empty()) {
            ROS_DEBUG_THROTTLE(1, "No detections in array");
            return;
        }

        // 选择离图像中心最近的检测
        const auto& best = *std::min_element(
            detections->detections.begin(), detections->detections.end(),
            [](const rm_radar_msgs::DroneDetection& a, const rm_radar_msgs::DroneDetection& b) {
                return a.distance_to_image_center < b.distance_to_image_center;
            });
        processSingleDetection(best);
    }

    void PoseSolver::processSingleDetection(const rm_radar_msgs::DroneDetection& detection)
    {
        armor_2d_points_.clear();
        
        for (const auto& pt : detection.armor_points) {
            armor_2d_points_.emplace_back(cv::Point2d(pt.x, pt.y));
        }

        if (intrinsics_.empty() || dist_coeffs_.empty()) {
            ROS_WARN("Camera parameters not available yet. Cannot solve pose.");
            return;
        }

        if(armor_2d_points_.size() != 4)
        {
            ROS_WARN("Expected 4 armor points, but got %lu. Cannot solve pose.", armor_2d_points_.size());
            return;
        }

        if(solvePose())
        {
            // 在优化之前，先计算去畸变点和相机参数
            getReprojectParams();
            
            optimizeYaw();
            
            // 发布求解后的位姿给Tracker
            rm_radar_msgs::DroneDetection solved_msg = detection;
            solved_msg.pose.position.x = tvec_(0);
            solved_msg.pose.position.y = tvec_(1);
            solved_msg.pose.position.z = tvec_(2);
            solved_msg.pose.orientation.x = optimized_q_.x();
            solved_msg.pose.orientation.y = optimized_q_.y();
            solved_msg.pose.orientation.z = optimized_q_.z();
            solved_msg.pose.orientation.w = optimized_q_.w();
            solved_msg.centroid.x = tvec_(0);
            solved_msg.centroid.y = tvec_(1);
            solved_msg.centroid.z = tvec_(2);
            solved_pose_pub_.publish(solved_msg);
            
            // Note: TF is published by Tracker, not here to avoid duplicate TF
            // publishTransform(detection);
            
            ROS_INFO_THROTTLE(1, "[PoseSolver] Published solved pose: (%.3f, %.3f, %.3f)", 
                tvec_(0), tvec_(1), tvec_(2));
        }
        else
        {
            ROS_WARN("Pose solving failed for track id %d", detection.id);
        }
    }

    void PoseSolver::publishTransform(const rm_radar_msgs::DroneDetection& detection)
    {
        std::string child_frame_id = "drone_" + std::to_string(detection.id);
        try
        {
            geometry_msgs::TransformStamped tf;
            tf.header = detection.header;
            tf.header.frame_id = detection.header.frame_id;
            tf.child_frame_id = child_frame_id;
            tf.transform.translation.x = tvec_(0);
            tf.transform.translation.y = tvec_(1);
            tf.transform.translation.z = tvec_(2);
            tf.transform.rotation.x = optimized_q_.x();
            tf.transform.rotation.y = optimized_q_.y();
            tf.transform.rotation.z = optimized_q_.z();
            tf.transform.rotation.w = optimized_q_.w();

            tf_broadcaster_->sendTransform(tf);
        }
        catch(tf2::TransformException& e)
        {
            ROS_ERROR("Failed to publish transform: %s", e.what());
        }

    }

} // namespace rm_radarplugin
