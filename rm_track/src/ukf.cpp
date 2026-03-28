#include "ukf.h"
namespace rm_radarplugin
{
    UKF::UKF(ros::NodeHandle& nh)
    {
        nh_ = nh;
        ROS_INFO("UKF constructor called.");
    }
    
    void UKF::initDynamicReconfigure()
    {
        if(ukf_cfg_srv_ != nullptr)
        {
            ROS_WARN("UKF dynamic_reconfigure server already exists!");
            return;
        }
        
        ukf_cfg_srv_ = new dynamic_reconfigure::Server<rm_track::ukfConfig>(ros::NodeHandle(nh_, "ukf"));
        dynamic_reconfigure::Server<rm_track::ukfConfig>::CallbackType f;
        f = boost::bind(&UKF::ukfconfigCB, this, _1, _2);
        ukf_cfg_srv_->setCallback(f);
        
        ROS_INFO("UKF dynamic_reconfigure server created.");
    }

    void UKF::initialize(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0)
    {
         // 1. 设置所有状态变量
         state_ = x0;    
         P_ = P0;        
         
         state_dim_ = static_cast<int>(x0.size()); //get dim
         sigma_point_count_ = 2 * state_dim_ + 1; // 2n+1个sigma点

        // Default UKF scaling if not set elsewhere
        lambda_ = 3.0 - state_dim_;

        // allocate sigma points and weights
        Xsig_.resize(state_dim_, sigma_point_count_);
        Xsig_pred_.resize(state_dim_, sigma_point_count_);
        weights_m_.resize(sigma_point_count_, 1);
        weights_c_.resize(sigma_point_count_, 1);

        //0，0中心点权重
        weights_m_(0, 0) = lambda_ / (lambda_ + state_dim_);
        weights_c_(0, 0) = weights_m_(0, 0);
        
        // 其他sigma点权重
        const double w = 0.5 / (lambda_ + state_dim_);
        for (int i = 1; i < sigma_point_count_; ++i) {
            weights_m_(i, 0) = w;
            weights_c_(i, 0) = w;
        }

        //噪声矩阵Q R初始化
        Q_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_) * 0.01;
        Q_base_ = Q_;  // 基础Q矩阵初始化
        meas_dim_ = 3;  // 默认观察三个维度 (x, y, z)
        R_ = Eigen::MatrixXd::Identity(meas_dim_, meas_dim_) * 0.1;
        R_base_ = R_;  // 基础R矩阵初始化

        // 2. 设置 ukf_initialized_ 为 true
        ukf_initialized_ = true;
         
        ROS_INFO("UKF State set. Dim: %d, meas_dim: %d", state_dim_, meas_dim_);

        // 3. 手动触发一次回调，使用当前配置更新 Q/R 矩阵
        if(ukf_cfg_srv_ != nullptr)
        {
            rm_track::ukfConfig current_config;
            ukf_cfg_srv_->getConfigDefault(current_config);
            
            ros::NodeHandle pnh(nh_, "ukf");
            pnh.param("model_type", current_config.model_type, current_config.model_type);
            pnh.param("q_pos", current_config.q_pos, current_config.q_pos);
            pnh.param("q_vel_xy", current_config.q_vel_xy, current_config.q_vel_xy);
            pnh.param("q_vel_z", current_config.q_vel_z, current_config.q_vel_z);
            pnh.param("q_acc_xy", current_config.q_acc_xy, current_config.q_acc_xy);
            pnh.param("q_acc_z", current_config.q_acc_z, current_config.q_acc_z);
            pnh.param("q_yaw", current_config.q_yaw, current_config.q_yaw);
            pnh.param("q_vyaw", current_config.q_vyaw, current_config.q_vyaw);
            pnh.param("q_r", current_config.q_r, current_config.q_r);
            pnh.param("q_dz", current_config.q_dz, current_config.q_dz);
            pnh.param("r_pos_xy", current_config.r_pos_xy, current_config.r_pos_xy);
            pnh.param("r_pos_z", current_config.r_pos_z, current_config.r_pos_z);
            pnh.param("r_yaw", current_config.r_yaw, current_config.r_yaw);
            pnh.param("debug_mode", current_config.debug_mode, current_config.debug_mode);
            
            ukfconfigCB(current_config, 0);
            ROS_INFO("UKF initialized with model_type=%d", current_config.model_type);
        }
     }

    void UKF::ukfconfigCB(rm_track::ukfConfig& config, uint32_t level)
    {   
        ROS_INFO("========== UKF Config Callback ==========");
        ROS_INFO("  debug_mode: %d -> %d", debug_mode_, config.debug_mode);
        ROS_INFO("  model_type: %d -> %d", model_type_, config.model_type);
        ROS_INFO("  q_pos: %.4f (raw: %.4f)", config.q_pos * config.q_pos, config.q_pos);
        ROS_INFO("  r_pos_xy: %.4f (raw: %.4f)", config.r_pos_xy * config.r_pos_xy, config.r_pos_xy);
        ROS_INFO("==========================================");
        
        //model type
        model_type_ = config.model_type;
        debug_mode_ = config.debug_mode;

        //Q var - store values regardless of initialization state
        q_pos_ = config.q_pos * config.q_pos;
        q_vel_xy_ = config.q_vel_xy * config.q_vel_xy;
        q_vel_z_ = config.q_vel_z * config.q_vel_z;
        q_acc_xy_ = config.q_acc_xy * config.q_acc_xy;
        q_acc_z_ = config.q_acc_z * config.q_acc_z;
        q_yaw_ = config.q_yaw * config.q_yaw;
        q_vyaw_ = config.q_vyaw * config.q_vyaw;
        q_r_ = config.q_r * config.q_r;
        q_dz_ = config.q_dz * config.q_dz;

        //R var
        r_pos_xy_ = config.r_pos_xy * config.r_pos_xy;
        r_pos_z_ = config.r_pos_z * config.r_pos_z;
        r_yaw_ = config.r_yaw * config.r_yaw;

        // Only update matrices if UKF is initialized
        if(!ukf_initialized_ || state_dim_ <= 0 || meas_dim_ <= 0)
        {
            ROS_WARN_THROTTLE(2, "UKF not initialized yet, skipping Q/R matrix update. state_dim=%d, meas_dim=%d", 
                              state_dim_, meas_dim_);
            return;
        }

        //matrix set up
        //Q - Resize Q_ to match state_dim_ if needed, then populate
        // Ensure Q_ has correct dimensions
        if(Q_.rows() != state_dim_ || Q_.cols() != state_dim_)
        {
            Q_ = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
        }
        else
        {
            Q_.setZero();
        }
        
        // 首先检查 model_type 与 state_dim 是否匹配
        bool model_state_mismatch = false;
        switch(model_type_)
        {
            case rm_track::CV: //[x, y, z, vx, vy, vz] (6维)
            {
                if(state_dim_ < 6)
                {
                    model_state_mismatch = true;
                    ROS_WARN_THROTTLE(5, "UKF: CV model requires 6D state, but state_dim=%d. Using fallback Q.", state_dim_);
                }
                else
                {
                    Q_.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * q_pos_;
                    Q_.block<2,2>(3,3) = Eigen::Matrix2d::Identity() * q_vel_xy_;
                    Q_(5,5) = q_vel_z_;
                }
                break;
            }
            case rm_track::CA: //[x, y, z, vx, vy, vz, ax, ay, az] (9维)
            {
                if(state_dim_ < 9)
                {
                    model_state_mismatch = true;
                    ROS_WARN_THROTTLE(5, "UKF: CA model requires 9D state, but state_dim=%d. Using fallback Q.", state_dim_);
                }
                else
                {
                    Q_.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * q_pos_;
                    Q_.block<2,2>(3,3) = Eigen::Matrix2d::Identity() * q_vel_xy_;
                    Q_(5,5) = q_vel_z_;
                    Q_.block<2,2>(6,6) = Eigen::Matrix2d::Identity() * q_acc_xy_;
                    Q_(8,8) = q_acc_z_;
                }
                break;
            }
            case rm_track::CTRV: //[x, y, z, v, yaw, yaw_rate] 或更高维
            {
                if(state_dim_ < 6)
                {
                    model_state_mismatch = true;
                    ROS_WARN_THROTTLE(5, "UKF: CTRV model requires at least 6D state, but state_dim=%d. Using fallback Q.", state_dim_);
                }
                else if(state_dim_ >= 9)
                {
                    Q_.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * q_pos_;
                    Q_.block<2,2>(3,3) = Eigen::Matrix2d::Identity() * q_vel_xy_;
                    Q_(5,5) = q_vel_z_;
                    Q_(6,6) = q_yaw_;
                    Q_(7,7) = q_vyaw_;
                    Q_(8,8) = q_r_;
                }
                else
                {
                    // 6D CTRV: [x, y, z, v, yaw, yaw_rate]
                    Q_.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * q_pos_;
                    Q_(3,3) = q_vel_xy_;  // v
                    Q_(4,4) = q_yaw_;
                    Q_(5,5) = q_vyaw_;
                }
                break;
            }
            default:
            {
                model_state_mismatch = true;
                ROS_WARN_THROTTLE(5, "UKF: Unknown model_type=%d, using fallback Q.", model_type_);
                break;
            }
        }
        
        // 如果 model/state 不匹配，使用通用的对角Q矩阵
        if(model_state_mismatch)
        {
            // 位置噪声用于前3维，其他用速度噪声
            for(int i = 0; i < state_dim_; i++)
            {
                if(i < 3)
                    Q_(i, i) = q_pos_;
                else
                    Q_(i, i) = q_vel_xy_;
            }
        }

        //R - Ensure R_ has correct dimensions
        if(R_.rows() != meas_dim_ || R_.cols() != meas_dim_)
        {
            R_ = Eigen::MatrixXd::Zero(meas_dim_, meas_dim_);
        }
        else
        {
            R_.setZero();
        }
        
        // Safe fill R diagonal - avoid segment() which can crash on edge cases
        for(int i = 0; i < std::min(meas_dim_, 2); i++)
        {
            R_(i, i) = r_pos_xy_;
        }
        if(meas_dim_ > 2)
        {
            R_(2,2) = r_pos_z_;
        }
        //yaw
        if (meas_dim_ > 3) 
        {
            R_(3, 3) = r_yaw_;
        }
        
        // 保存基础矩阵，供AIMM/Tracker自适应缩放使用
        Q_base_ = Q_;
        R_base_ = R_;
        
        ROS_INFO("UKF Q/R matrices updated. Q diag: [%.4f, %.4f, %.4f, ...], R diag: [%.4f, %.4f, %.4f]",
                          Q_(0,0), Q_(1,1), Q_(2,2), R_(0,0), R_(1,1), meas_dim_ > 2 ? R_(2,2) : 0.0);
        
        if(debug_mode_)
        {
            printQRMatrices();
        }
    }

    void UKF::printQRMatrices()
    {
        if(!ukf_initialized_ || state_dim_ <= 0 || meas_dim_ <= 0)
        {
            ROS_WARN("[UKF Debug] Cannot print Q/R - UKF not initialized");
            return;
        }
        
        ROS_INFO("============ UKF DEBUG: Q Matrix (%dx%d) ============", state_dim_, state_dim_);
        std::stringstream ss_q;
        ss_q << std::fixed << std::setprecision(6);
        for(int i = 0; i < state_dim_; i++)
        {
            ss_q << "  [";
            for(int j = 0; j < state_dim_; j++)
            {
                ss_q << std::setw(12) << Q_(i, j);
                if(j < state_dim_ - 1) ss_q << ", ";
            }
            ss_q << "]";
            ROS_INFO("%s", ss_q.str().c_str());
            ss_q.str("");
        }
        
        ROS_INFO("============ UKF DEBUG: R Matrix (%dx%d) ============", meas_dim_, meas_dim_);
        std::stringstream ss_r;
        ss_r << std::fixed << std::setprecision(6);
        for(int i = 0; i < meas_dim_; i++)
        {
            ss_r << "  [";
            for(int j = 0; j < meas_dim_; j++)
            {
                ss_r << std::setw(12) << R_(i, j);
                if(j < meas_dim_ - 1) ss_r << ", ";
            }
            ss_r << "]";
            ROS_INFO("%s", ss_r.str().c_str());
            ss_r.str("");
        }
        
        ROS_INFO("============ UKF DEBUG: Current State ============");
        if(state_.size() >= 6)
        {
            ROS_INFO("  State: [x=%.4f, y=%.4f, z=%.4f, vx=%.4f, vy=%.4f, vz=%.4f%s]",
                     state_(0), state_(1), state_(2), state_(3), state_(4), state_(5),
                     state_.size() > 6 ? ", ..." : "");
        }
        ROS_INFO("==================================================");
    }

    void UKF::printQRMatricesThrottled()
    {
        static ros::Time last_print_time = ros::Time(0);
        ros::Time now = ros::Time::now();
        
        if((now - last_print_time).toSec() < 2.0)
        {
            return;
        }
        last_print_time = now;
        
        printQRMatrices();
    }



    void UKF::generateSigmaPoints()
    {
        // 确保P_是对称的
        P_ = 0.5 * (P_ + P_.transpose());
        
        //// P = A * A^T, Cholesky Decomposition
        Eigen::LLT<Eigen::MatrixXd> lltOfP(P_);
        
        int max_attempts = 5;
        int attempt = 0;
        while(lltOfP.info() == Eigen::NumericalIssue && attempt < max_attempts)
        {
            attempt++;
            // 添加更大的jitter
            double jitter = 1e-6 * std::pow(10, attempt);
            P_ += jitter * Eigen::MatrixXd::Identity(P_.rows(), P_.cols());
            P_ = 0.5 * (P_ + P_.transpose());  // 保持对称
            lltOfP.compute(P_);
            
            if(attempt == 1)
            {
                ROS_WARN_THROTTLE(2, "UKF: LLT failed, adding jitter (attempt %d, jitter=%.2e)", attempt, jitter);
            }
        }
        
        if(lltOfP.info() == Eigen::NumericalIssue)
        {
            ROS_ERROR_THROTTLE(2, "UKF: LLT still failed after %d attempts, resetting P to identity", max_attempts);
            P_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_) * 0.1;
            lltOfP.compute(P_);
        }

        Eigen::MatrixXd A = lltOfP.matrixL();

        Xsig_.col(0) = state_;
        double scale = sqrt(lambda_ + state_dim_);

        for (int i = 0; i < state_dim_; i++)
        {
            Xsig_.col(i + 1) = state_ + scale * A.col(i);
            Xsig_.col(i + 1 + state_dim_) = state_ - scale * A.col(i);
        }
    }

    Eigen::VectorXd UKF::processModel(const Eigen::VectorXd& x)
    {
        Eigen::VectorXd x_pred = x;

        switch(model_type_)
        {
            //ConstantVelocity
            case rm_track::CV:
            {
                ROS_INFO_ONCE("Using CV model");
                // [px, py, pz, vx, vy, vz] (6维)
                if (x.size() < 6) {
                    ROS_ERROR("UKF: State vector too small for CV 3D model!,prediction skipped");
                    return x;
                }

                // 位置 = 位置 + 速度 * dt_
                x_pred(0) += x(3) * dt_; // px
                x_pred(1) += x(4) * dt_; // py
                x_pred(2) += x(5) * dt_; // pz
                
                // 速度保持不变 (vx, vy, vz)
                break;
            }

            //ConstantAcceleration
            case rm_track::CA:
            {
                ROS_INFO_ONCE("Using CA model");
                //[px, py, pz, vx, vy, vz, ax, ay, az] (9维)
                if (x.size() < 9) {
                    ROS_ERROR("UKF: State vector too small for CA 3D model!,prediction skipped");
                    return x;
                }

                double dt_2 = 0.5 * dt_ * dt_;
                
                // 位置
                x_pred(0) += x(3) * dt_ + x(6) * dt_2; // px
                x_pred(1) += x(4) * dt_ + x(7) * dt_2; // py
                x_pred(2) += x(5) * dt_ + x(8) * dt_2; // pz

                // 速度
                x_pred(3) += x(6) * dt_; // vx
                x_pred(4) += x(7) * dt_; // vy
                x_pred(5) += x(8) * dt_; // vz

                // 加速度保持不变 (ax, ay, az)
                break;
            }
            
            //ConstantTurnRateAndVelocity
            case rm_track::CTRV:
            {
                ROS_INFO_ONCE("Using CTRV model");
                // [px, py, pz, v, yaw, yaw_rate] (6维)
                if(x.size() < 6) {
                    ROS_ERROR("UKF: State vector too small for CTRV 3D model!,prediction skipped");
                    return x;
                }

                double px = x(0);
                double py = x(1);
                double pz = x(2);
                double v = x(3);
                double yaw = x(4);
                double yaw_rate = x(5);

                if (fabs(yaw_rate) > 1e-5)
                {
                    x_pred(0) = px + (v / yaw_rate) * (sin(yaw + yaw_rate * dt_) - sin(yaw));
                    x_pred(1) = py + (v / yaw_rate) * (-cos(yaw + yaw_rate * dt_) + cos(yaw));
                }
                else
                {
                    x_pred(0) = px + v * cos(yaw) * dt_;
                    x_pred(1) = py + v * sin(yaw) * dt_;
                }
                
                x_pred(2) = pz; 

                x_pred(4) = yaw + yaw_rate * dt_; // yaw 更新
                
                break;
            }
        }

        return x_pred;
    }

    Eigen::VectorXd UKF::computeMean(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& sigma_points, int dim)
    {
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
        for (int i = 0; i < sigma_point_count_; i++)
        {
            mean += weights(i) * sigma_points.col(i);
        }
        return mean;

    }

    Eigen::MatrixXd UKF::computeCovariance(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& sigma_points, const Eigen::VectorXd& mean, int dim, int angle_idx)
    {
        Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(dim, dim);
        for (int i = 0; i < sigma_point_count_; i++)
        {
            Eigen::VectorXd diff = sigma_points.col(i) - mean;

            // 只有当 angle_idx 有效时才归一化
            if(angle_idx >= 0 && angle_idx < dim) {
                normalizeAngle(diff(angle_idx));
            }
            
            covariance += weights_c_(i) * (diff * diff.transpose());
        }
        return covariance;
    }



    void UKF::predict()
    {
        //1. generate sigma points
        generateSigmaPoints();

        //2. predict sigma points
        for(int i = 0; i < sigma_point_count_; i++)
        {
            Xsig_pred_.col(i) = processModel(Xsig_.col(i));
        }

        //3. compute predicted mean and covariance
        Eigen::VectorXd state_pred = computeMean(weights_m_, Xsig_pred_, state_dim_);
        
        // 对于 CTRV 模型，yaw 角在索引4，需要进行角度归一化
        int angle_idx = (model_type_ == rm_track::CTRV) ? 4 : -1;
        Eigen::MatrixXd P_pred = computeCovariance(weights_c_, Xsig_pred_, state_pred, state_dim_, angle_idx);
        
        //4. add process noise Q to covariance
        P_pred += Q_;

        //5. update state
        state_ = state_pred;
        P_ = P_pred;

        // Debug output (throttled)
        if(debug_mode_)
        {
            printQRMatricesThrottled();
        }
    }

    void UKF::update(const Eigen::VectorXd& z_meas)
    {
        int meas_dim = z_meas.size();

        //1. predict measurement sigma points
        Zsig_ = Eigen::MatrixXd::Zero(meas_dim, sigma_point_count_);
        for(int i = 0; i < sigma_point_count_; i++)
        {
            //测量模型，这里假设直接观测状态量
            Zsig_.col(i) = Xsig_pred_.col(i).head(meas_dim);
        }

        //2. compute predicted measurement mean and covariance
        z_pred_ = computeMean(weights_m_, Zsig_, meas_dim);
        S_ = computeCovariance(weights_c_, Zsig_, z_pred_, meas_dim_, -1);
        //3. add measurement noise R to covariance
        S_ += R_;

        //4. compute cross-correlation matrix
        Tc_ = Eigen::MatrixXd::Zero(state_dim_, meas_dim);
        for(int i = 0; i < sigma_point_count_; i++)
        {
            Eigen::VectorXd x_diff = Xsig_pred_.col(i) - state_;
            if(model_type_ == rm_track::CTRV && x_diff.size() > 4) 
            {
                normalizeAngle(x_diff(4));
            }

            Eigen::VectorXd z_diff = Zsig_.col(i) - z_pred_;
            Tc_ += weights_c_(i) * (x_diff * z_diff.transpose());
        }

        //5. compute Kalman gain K
        K_ = Tc_ * S_.inverse();

        //6. update state and covariance
        Eigen::VectorXd z_diff = z_meas - z_pred_;
        state_ += K_ * z_diff;
        P_ -= K_ * S_ * K_.transpose();

        //7. 确保P矩阵保持正定和对称
        // 强制对称
        P_ = 0.5 * (P_ + P_.transpose());
        
        // 检查并修复对角线元素不能为负或太小
        for(int i = 0; i < state_dim_; i++)
        {
            if(P_(i, i) < 1e-6)
            {
                P_(i, i) = 1e-6;
            }
        }

        if(debug_mode_)
        {
            // Normalized Innovation Squared
            NIS_ = z_diff.transpose() * S_.inverse() * z_diff;
            // print NIS value
            ROS_INFO_THROTTLE(1, "UKF Update: meas_dim=%d, NIS=%.4f", meas_dim, NIS_);
        }
    }

    Eigen::VectorXd UKF::getState() const
    {
        return state_;
    }

    Eigen::MatrixXd UKF::getCovariance() const
    {
        return P_;
    }

    double UKF::computeInnovation(const Eigen::VectorXd& z_meas,
                                   Eigen::VectorXd& z_pred_out,
                                   Eigen::MatrixXd& S_out) const
    {
        int meas_dim = z_meas.size();

        // Map predicted sigma points into measurement space
        Eigen::MatrixXd Zsig = Eigen::MatrixXd::Zero(meas_dim, sigma_point_count_);
        for (int i = 0; i < sigma_point_count_; i++)
        {
            Zsig.col(i) = Xsig_pred_.col(i).head(meas_dim);
        }

        // Predicted measurement mean (using sigma-point weights)
        z_pred_out = Eigen::VectorXd::Zero(meas_dim);
        for (int i = 0; i < sigma_point_count_; i++)
        {
            z_pred_out += weights_m_(i) * Zsig.col(i);
        }

        // Innovation covariance S = Σ w_c * (z_sig - z_pred)(z_sig - z_pred)^T + R
        S_out = Eigen::MatrixXd::Zero(meas_dim, meas_dim);
        for (int i = 0; i < sigma_point_count_; i++)
        {
            Eigen::VectorXd z_diff = Zsig.col(i) - z_pred_out;
            S_out += weights_c_(i) * (z_diff * z_diff.transpose());
        }
        S_out += R_;

        // Innovation
        Eigen::VectorXd innovation = z_meas - z_pred_out;

        // NIS = ν^T S^{-1} ν
        double nis = innovation.transpose() * S_out.inverse() * innovation;
        return std::max(nis, 0.0);
    }

}  // namespace rm_radarplugin


