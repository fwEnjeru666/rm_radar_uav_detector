#include "ukf.h"
namespace rm_radarplugin
{
    UKF::UKF(ros::NodeHandle& nh, std::shared_ptr<BaseModel> model)
    : model_(model)
    {   
        if(model_)
        {
            weightsInit(model_);
            ROS_INFO("UKF constructor called with model. State dim: %d", state_dim_);
        }
        nh_ = nh;

        R_ = Eigen::MatrixXd::Identity(3, 3);
        // Setup Dynamic Reconfigure Server
        ukf_cfg_srv_ = std::make_unique<dynamic_reconfigure::Server<rm_track::ukfConfig>>(nh_);
        ukf_cfg_cb_ = boost::bind(&UKF::ukfconfigCB, this, _1, _2);
        ukf_cfg_srv_->setCallback(ukf_cfg_cb_);
    }

    void UKF::setModel(std::shared_ptr<BaseModel> model)
    {
        if (!model)
        {
            ROS_ERROR("[UKF::setModel] New model is null, keeping current model.");
            return;
        }

        model_ = model;
        weightsInit(model_);

        // Model switch changes state dimension. Force re-initialization on next measurement.
        ukf_initialized_ = false;
        state_.resize(0);
        P_.resize(0, 0);
        Xsig_.resize(0, 0);
        Xsig_pred_.resize(0, 0);
        Zsig_.resize(0, 0);
        z_pred_.resize(0);
        z_meas_.resize(0);
        S_.resize(0, 0);
        Tc_.resize(0, 0);
        K_.resize(0, 0);

        // Reset measurement-noise storage to default 3D layout until next initialize/update.
        meas_dim_ = 3;
        R_ = Eigen::MatrixXd::Identity(3, 3);

        ROS_INFO("[UKF::setModel] Switched model to %s (state_dim=%d). Filter will reinitialize.",
                 model_->getName().c_str(), state_dim_);
    }

    void UKF::weightsInit(std::shared_ptr<BaseModel> model)
    {
        state_dim_ = model->get_QMatrix().rows();
        meas_dim_ = 3;
        lambda_ = 3.0 - state_dim_;
        sigma_point_count_ = 2 * state_dim_ + 1; // 2n+1个sigma点

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

        ROS_INFO("UKF weights initialized. State dim: %d, Sigma points: %d", state_dim_, sigma_point_count_);
    }

    void UKF::initialize(const Eigen::Vector3d& z_meas)
    {
        if(!model_)
        {
            ROS_ERROR("UKF initialization failed: model is null");
            return;
        }
        ModelState init_state = model_->generateInitialState(z_meas);
        state_ = init_state.state;
        P_ = init_state.covariance;

        Xsig_.resize(state_dim_, sigma_point_count_);
        Xsig_pred_.resize(state_dim_, sigma_point_count_);
        
        // 初始化时就把 Zsig 的内存分好，防止后续空指
        Zsig_.resize(meas_dim_ > 0 ? meas_dim_ : 3, sigma_point_count_);

        ukf_initialized_ = true;
        ROS_INFO("UKF initialized with measurement (%.3f, %.3f, %.3f)", z_meas(0), z_meas(1), z_meas(2));
    }

    void UKF::ukfconfigCB(rm_track::ukfConfig& config, uint32_t level)
    {   
        ROS_INFO("========== UKF Config Callback ==========");
        ROS_INFO("  debug_mode: %d -> %d", debug_mode_, config.debug_mode);
        ROS_INFO("==========================================");
        
        debug_mode_ = config.debug_mode;

        // Only update matrices if UKF is initialized
        if(!ukf_initialized_ || state_dim_ <= 0 || meas_dim_ <= 0)
        {
            ROS_WARN_THROTTLE(2, "UKF not initialized yet, skipping Q/R matrix update. state_dim=%d, meas_dim=%d", 
                              state_dim_, meas_dim_);
            return;
        }
        
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

    Eigen::VectorXd UKF::computeMean(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& sigma_points, int dim) const
    {
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
        for (int i = 0; i < sigma_point_count_; ++i)
        {
            mean += weights(i) * sigma_points.col(i);
        }
        return mean;
    }

    Eigen::MatrixXd UKF::computeCovariance(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& sigma_points, const Eigen::VectorXd& mean, int dim, int angle_idx) const
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
        if (!ukf_initialized_ || !model_)
        {
            ROS_WARN_THROTTLE(2, "[UKF::predict] Not initialized or model is null. Skipping.");
            return;
        }

        //1. generate sigma points
        generateSigmaPoints();

        //2. predict sigma points
        for(int i = 0; i < sigma_point_count_; i++)
        {
            Xsig_pred_.col(i) = model_->processFunction(Xsig_.col(i), dt_);
        }
        
        //3. compute predicted mean and covariance
        state_ = computeMean(weights_m_, Xsig_pred_, state_dim_);
       
        //4. add process noise Q to covariance
        int angle_idx = model_->getStateAngleIdx();
        P_ = computeCovariance(weights_c_, Xsig_pred_, state_, state_dim_, angle_idx);
        P_ += model_->get_QMatrix();
        Zsig_.resize(meas_dim_, sigma_point_count_);
        for(int i = 0; i < sigma_point_count_; i++)
        {
            Zsig_.col(i) = model_->measurementFunction(Xsig_pred_.col(i), meas_dim_);
        }

        if(debug_mode_)
        {
            printQRMatricesThrottled();
        }
    }

    void UKF::update(const Eigen::VectorXd& z_meas)
    {
        if(!ukf_initialized_)
        {
            ROS_WARN_THROTTLE(2, "UKF update called before initialization!");
            return;
        }

        int meas_dim = z_meas.size();

        // Handle runtime measurement-dimension switching (e.g. 3D <-> 8D)
        if (meas_dim_ != meas_dim || Zsig_.rows() != meas_dim)
        {
            ROS_WARN_THROTTLE(1, "[UKF::update] Adjusting measurement dim from %d to %d", meas_dim_, meas_dim);
            meas_dim_ = meas_dim;
            Zsig_.resize(meas_dim_, sigma_point_count_);
            for (int i = 0; i < sigma_point_count_; ++i)
            {
                Zsig_.col(i) = model_->measurementFunction(Xsig_pred_.col(i), meas_dim_);
            }
        }

        if (Zsig_.cols() != sigma_point_count_)
        {
            ROS_ERROR_THROTTLE(2, "[UKF::update] Zsig_ is empty! Did you forget to call predict() first?");
            return;
        }

        if (R_.rows() != meas_dim || R_.cols() != meas_dim)
        {
            ROS_WARN_THROTTLE(1, "[UKF::update] R size (%d,%d) mismatches meas_dim=%d, resetting to identity",
                              static_cast<int>(R_.rows()), static_cast<int>(R_.cols()), meas_dim);
            R_ = Eigen::MatrixXd::Identity(meas_dim, meas_dim);
        }

        //1. compute predicted measurement mean and covariance
        z_pred_ = computeMean(weights_m_, Zsig_, meas_dim);
        S_ = computeCovariance(weights_c_, Zsig_, z_pred_, meas_dim, -1);
        
        //2. add measurement noise R to covariance
        S_ += R_;

        //3. compute cross-correlation matrix
        Tc_ = Eigen::MatrixXd::Zero(state_dim_, meas_dim);
        int angle_idx = model_->getStateAngleIdx();

        for(int i = 0; i < sigma_point_count_; i++)
        {
            Eigen::VectorXd x_diff = Xsig_pred_.col(i) - state_;
            if(angle_idx >= 0 && x_diff.size() > angle_idx) 
            {
                normalizeAngle(x_diff(angle_idx));
            }
            Eigen::VectorXd z_diff = Zsig_.col(i) - z_pred_;
            Tc_ += weights_c_(i) * (x_diff * z_diff.transpose());
        }

        //4. compute Kalman gain K
        Eigen::LLT<Eigen::MatrixXd> llt_S(S_);
        
        if (llt_S.info() == Eigen::NumericalIssue) 
        {
            ROS_WARN_THROTTLE(1, "[UKF::update] S matrix is singular! Adding jitter.");
            S_ += Eigen::MatrixXd::Identity(meas_dim, meas_dim) * 1e-3;
            llt_S.compute(S_);
            if (llt_S.info() == Eigen::NumericalIssue) {
                ROS_FATAL_THROTTLE(1, "[UKF::update] S matrix is STILL singular! Aborting update.");
                return; 
            }
        }
        
        K_ = Tc_ * llt_S.solve(Eigen::MatrixXd::Identity(meas_dim, meas_dim));

        //5. update state and covariance
        Eigen::VectorXd z_diff = z_meas - z_pred_;
        
        state_ += K_ * z_diff;
        
        if (angle_idx >= 0 && state_.size() > angle_idx)
        {
            normalizeAngle(state_(angle_idx));
        }
        
        P_ -= K_ * S_ * K_.transpose();

        //6. 确保P矩阵保持正定和对称
        P_ = 0.5 * (P_ + P_.transpose());
        
        for(int i = 0; i < state_dim_; i++)
        {
            if(P_(i, i) < 1e-6) P_(i, i) = 1e-6;
        }

        if(debug_mode_)
        {
            NIS_ = z_diff.transpose() * llt_S.solve(z_diff);
            ROS_INFO_THROTTLE(1, "UKF Update: meas_dim=%d, NIS=%.4f", meas_dim, NIS_);
        }
    }

    double UKF::computeInnovation(const Eigen::VectorXd& z_meas,
                                   Eigen::VectorXd& z_pred_out,
                                   Eigen::MatrixXd& S_out) const
    {
        if (!ukf_initialized_ || Zsig_.cols() != sigma_point_count_)
        {
            ROS_WARN_THROTTLE(2, "[UKF::computeInnovation] Filter not ready. Returning massive NIS.");
            z_pred_out = Eigen::VectorXd::Zero(z_meas.size());
            S_out = Eigen::MatrixXd::Identity(z_meas.size(), z_meas.size());
            return 1000.0; 
        }

        int meas_dim = z_meas.size();

        Eigen::MatrixXd Zsig_local;
        const Eigen::MatrixXd* zsig_ptr = &Zsig_;
        if (Zsig_.rows() != meas_dim)
        {
            Zsig_local.resize(meas_dim, sigma_point_count_);
            for (int i = 0; i < sigma_point_count_; ++i)
            {
                Zsig_local.col(i) = model_->measurementFunction(Xsig_pred_.col(i), meas_dim);
            }
            zsig_ptr = &Zsig_local;
        }

        Eigen::MatrixXd R_used = R_;
        if (R_used.rows() != meas_dim || R_used.cols() != meas_dim)
        {
            R_used = Eigen::MatrixXd::Identity(meas_dim, meas_dim);
        }

        z_pred_out = computeMean(weights_m_, *zsig_ptr, meas_dim);
        S_out = computeCovariance(weights_c_, *zsig_ptr, z_pred_out, meas_dim, -1) + R_used;

        Eigen::LLT<Eigen::MatrixXd> llt_S(S_out);
        if (llt_S.info() == Eigen::NumericalIssue)
        {
            S_out += Eigen::MatrixXd::Identity(meas_dim, meas_dim) * 1e-4;
            llt_S.compute(S_out);
        }

        Eigen::VectorXd innovation = z_meas - z_pred_out;
        
        int angle_idx = model_->getStateAngleIdx();
        if(angle_idx >= 0 && innovation.size() > angle_idx) 
        {
            normalizeAngle(innovation(angle_idx));
        }

        // 算出 NIS，如果依然解不开，返回一个默认的大值而不是 NaN
        double nis;
        if (llt_S.info() == Eigen::NumericalIssue) {
            nis = 1000.0;
        } else {
            nis = innovation.transpose() * llt_S.solve(innovation);
        }
        
        return std::max(nis, 0.0);
    }

    Eigen::VectorXd UKF::getState() const
    {
        return state_;
    }

    Eigen::MatrixXd UKF::getCovariance() const
    {
        return P_;
    }

}  // namespace rm_radarplugin