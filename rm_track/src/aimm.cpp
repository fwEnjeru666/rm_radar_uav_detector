#include "aimm.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace rm_radarplugin
{
    constexpr int AIMM::NUM_MODELS;
    constexpr int AIMM::OUTPUT_DIM;
    constexpr double AIMM::LAMBDA_REF;
    constexpr double AIMM::LIKELIHOOD_FLOOR;

    AIMM::AIMM(ros::NodeHandle& nh)
        : nh_(nh)
    {
        // Define model types
        model_types_ = {rm_track::CV, rm_track::CA, rm_track::CTRV};
        model_names_ = {"CV", "CA", "CTRV"};

        // Create one UKF per model, each with a unique sub-namespace
        filters_.reserve(NUM_MODELS);
        for (int i = 0; i < NUM_MODELS; ++i)
        {
            ros::NodeHandle model_nh(nh_, "aimm_" + model_names_[i]);
            filters_.emplace_back(model_nh);
        }

        // Initial model probabilities (uniform)
        mu_ = Eigen::VectorXd::Constant(NUM_MODELS, 1.0 / NUM_MODELS);

        // Initial TPM — will be overwritten every step by adaptTPM()
        TPM_ = Eigen::MatrixXd::Zero(NUM_MODELS, NUM_MODELS);
        double p_stay = p_stay_nominal_;
        double p_switch = (1.0 - p_stay) / (NUM_MODELS - 1);
        for (int i = 0; i < NUM_MODELS; ++i)
        {
            for (int j = 0; j < NUM_MODELS; ++j)
            {
                TPM_(i, j) = (i == j) ? p_stay : p_switch;
            }
        }
        
        // Allocate mixing storage
        mixed_states_.resize(NUM_MODELS);
        mixed_covs_.resize(NUM_MODELS);
        mixing_probs_ = Eigen::MatrixXd::Zero(NUM_MODELS, NUM_MODELS);

        // Adaptive storage
        Q_base_.resize(NUM_MODELS);
        model_NIS_.resize(NUM_MODELS, 0.0);
        // Initialize λ to the nominal value (ratio = 1.0 means no maneuver)
        lambda_ = LAMBDA_REF;
        // NIS baseline 首帧时一步校准，不预设值
        nis_baseline_ = 0.0;
        nis_baseline_initialized_ = false;

        // Output
        combined_state_ = Eigen::VectorXd::Zero(OUTPUT_DIM);
        combined_cov_ = Eigen::MatrixXd::Identity(OUTPUT_DIM, OUTPUT_DIM);

        // Read adaptive parameters from parameter server
        // nh_.param("aimm/lambda_ema_alpha", lambda_ema_alpha_, 0.3);
        // nh_.param("aimm/p_stay_min", p_stay_min_, 0.60);
        // nh_.param("aimm/p_stay_max", p_stay_max_, 0.98);
        // nh_.param("aimm/p_stay_nominal", p_stay_nominal_, 0.90);
        // nh_.param("aimm/q_scale_max", q_scale_max_, 10.0);

        // Setup Dynamic Reconfigure Server
        ros::NodeHandle aimm_nh(nh_, "aimm");
        aimm_cfg_server_ = std::make_shared<dynamic_reconfigure::Server<rm_track::aimmConfig>>(aimm_nh);
        dynamic_reconfigure::Server<rm_track::aimmConfig>::CallbackType cb = 
            boost::bind(&AIMM::aimmconfigCB, this, _1, _2);
        aimm_cfg_server_->setCallback(cb);

        ROS_INFO("[AIMM] Created with %d models: CV, CA, CTRV", NUM_MODELS);
        ROS_INFO("[AIMM] Adaptive params: ema=%.2f, p_stay=[%.2f,%.2f], q_scale_max=%.1f",
                 lambda_ema_alpha_, p_stay_min_, p_stay_max_, q_scale_max_);
    }

    void AIMM::aimmconfigCB(rm_track::aimmConfig& config, uint32_t level)
    {
        //ema
        lambda_ema_alpha_ = config.lambda_ema_alpha;
        nis_baseline_alpha_ = config.nis_baseline_alpha;
        
        //adaptTPM
        sigmoid_k_ = config.sigmoid_k;
        maneuver_gain_ = config.maneuver_gain;

        // Update TPM parameters
        p_stay_min_ = config.p_stay_min;
        p_stay_max_ = config.p_stay_max;
        p_stay_nominal_ = config.p_stay_nominal;
        q_scale_max_ = config.q_scale_max;
        
        ROS_INFO("[AIMM::Config] Parameters updated - ema: %.2f, p_stay_min: %.2f, p_stay_max: %.2f, p_stay_nom: %.2f, q_scale_max: %.1f",
                 lambda_ema_alpha_, p_stay_min_, p_stay_max_, p_stay_nominal_, q_scale_max_);
    }

    void AIMM::initDynamicReconfigure()
    {
        // Initialize dynamic reconfigure for all filters
        // They share the same config but each filter uses its own model_type
        for (auto& f : filters_)
        {
            f.initDynamicReconfigure();
        }
    }

    void AIMM::initialize(const Eigen::Vector3d& z_meas)
    {
        for (int i = 0; i < NUM_MODELS; ++i)
        {
            int model_type = model_types_[i];
            int state_dim = UKF::getStateDimForModel(model_type);

            Eigen::VectorXd x0 = Eigen::VectorXd::Zero(state_dim);
            Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(state_dim, state_dim);

            // Position from measurement
            x0.head(3) = z_meas;

            switch (model_type)
            {
                case rm_track::CV:  // [x, y, z, vx, vy, vz]
                    x0.tail(3).setZero();
                    P0.topLeftCorner(3, 3) *= 0.1;
                    P0.bottomRightCorner(3, 3) *= 10.0;
                    break;

                case rm_track::CA:  // [x, y, z, vx, vy, vz, ax, ay, az]
                    x0.segment(3, 3).setZero();
                    x0.segment(6, 3).setZero();
                    P0.block<3, 3>(0, 0) *= 0.1;
                    P0.block<3, 3>(3, 3) *= 10.0;
                    P0.block<3, 3>(6, 6) *= 1.0;
                    break;

                case rm_track::CTRV:  // [x, y, z, v, yaw, yaw_rate]
                    x0(3) = 0.0;  // v
                    x0(4) = 0.0;  // yaw
                    x0(5) = 0.0;  // yaw_rate
                    P0.topLeftCorner(3, 3) *= 0.1;
                    P0(3, 3) = 10.0;
                    P0(4, 4) = 0.5;
                    P0(5, 5) = 0.1;
                    break;
            }

            filters_[i].initialize(x0, P0);

            // Capture the base Q matrix from dynamic_reconfigure for adaptive scaling
            Q_base_[i] = filters_[i].getBaseProcessNoise();
            model_NIS_[i] = 0.0;

            ROS_INFO("[AIMM] Model %s initialized (state_dim=%d, Q_pos=%.6f, R_xy=%.6f)", 
                    model_names_[i].c_str(), state_dim,
                    Q_base_[i](0,0),
                    filters_[i].getMeasurementNoise()(0,0));
        }

        // Sanity check: warn if Q_pos >> R, which makes NIS perpetually tiny
        {
            double q0 = Q_base_[0](0,0);
            double r0 = filters_[0].getMeasurementNoise()(0,0);
            double ratio = q0 / std::max(r0, 1e-12);
            if (ratio > 100.0)
            {
                ROS_WARN("[AIMM] Q_pos/R_xy ratio = %.0f — Q is much larger than R. "
                         "S will be dominated by P (from Q), making NIS perpetually tiny. "
                         "Consider reducing q_pos (currently sqrt=%.3f) or increasing r_pos_xy.",
                         ratio, std::sqrt(q0));
            }
        }

        // Reset probabilities to uniform
        mu_ = Eigen::VectorXd::Constant(NUM_MODELS, 1.0 / NUM_MODELS);

        // Reset maneuver indicator to nominal (ratio = 1.0)
        lambda_ = LAMBDA_REF;
        nis_baseline_ = 0.0;
        nis_baseline_initialized_ = false;

        // Reset TPM to nominal
        double p_stay = p_stay_nominal_;
        double p_switch = (1.0 - p_stay) / (NUM_MODELS - 1);
        for (int i = 0; i < NUM_MODELS; ++i)
            for (int j = 0; j < NUM_MODELS; ++j)
                TPM_(i, j) = (i == j) ? p_stay : p_switch;

        // Set combined state
        combined_state_ = Eigen::VectorXd::Zero(OUTPUT_DIM);
        combined_state_.head(3) = z_meas;
        combined_cov_ = Eigen::MatrixXd::Identity(OUTPUT_DIM, OUTPUT_DIM) * 0.1;

        initialized_ = true;
        ROS_INFO("[AIMM] Initialized at (%.3f, %.3f, %.3f)", z_meas.x(), z_meas.y(), z_meas.z());
    }

    void AIMM::setDt(double dt)
    {
        for (auto& f : filters_)
        {
            f.setDt(dt);
        }
    }

    void AIMM::setMeasurementNoise(const Eigen::MatrixXd& R)
    {
        for (auto& f : filters_)
        {
            f.setMeasurementNoise(R);
        }
    }

    Eigen::MatrixXd AIMM::getMeasurementNoise() const
    {
        if (!filters_.empty())
            return filters_[0].getMeasurementNoise();
        return Eigen::Matrix3d::Identity() * 0.1;
    }

    Eigen::MatrixXd AIMM::getBaseMeasurementNoise() const
    {
        if (!filters_.empty())
            return filters_[0].getBaseMeasurementNoise();
        return Eigen::Matrix3d::Identity() * 0.1;
    }

    // ============================================================
    // Map model-specific state → unified 6D [x,y,z,vx,vy,vz]
    // ============================================================
    Eigen::VectorXd AIMM::toUnifiedState(const Eigen::VectorXd& state, int model_type) const
    {
        Eigen::VectorXd unified = Eigen::VectorXd::Zero(OUTPUT_DIM);
        
        switch (model_type)
        {
            case rm_track::CV:  // [x,y,z,vx,vy,vz] → direct copy
                unified = state.head(OUTPUT_DIM);
                break;

            case rm_track::CA:  // [x,y,z,vx,vy,vz,ax,ay,az] → take first 6
                unified = state.head(OUTPUT_DIM);
                break;

            case rm_track::CTRV:  // [x,y,z,v,yaw,yaw_rate] → decompose v into vx,vy
            {
                unified(0) = state(0);  // x
                unified(1) = state(1);  // y
                unified(2) = state(2);  // z
                double v = state(3);
                double yaw = state(4);
                unified(3) = v * std::cos(yaw);  // vx
                unified(4) = v * std::sin(yaw);  // vy
                unified(5) = 0.0;                // vz (CTRV doesn't model vz)
                break;
            }
            default:
                unified = state.head(std::min((int)state.size(), OUTPUT_DIM));
                break;
        }
        return unified;
    }

    // ============================================================
    // Helper: model covariance -> unified covariance (6D)
    // unified_x = [x, y, z, vx, vy, vz]
    // ============================================================
    static Eigen::MatrixXd toUnifiedCovariance(const Eigen::VectorXd& x_model,
                                               const Eigen::MatrixXd& P_model,
                                               int model_type)
    {
        constexpr int OUT = 6;
        Eigen::MatrixXd P_uni = Eigen::MatrixXd::Zero(OUT, OUT);

        if (P_model.rows() == 0 || P_model.cols() == 0)
        {
            P_uni.setIdentity();
            return P_uni;
        }

        switch (model_type)
        {
            case rm_track::CV:
            case rm_track::CA:
            {
                // CV/CA: first 6 states are already [x,y,z,vx,vy,vz]
                int dim = std::min<int>(P_model.rows(), OUT);
                P_uni.topLeftCorner(dim, dim) = P_model.topLeftCorner(dim, dim);
                break;
            }
            case rm_track::CTRV:
            {
                const int CTRV_DIM = std::min<int>(6, (int)P_model.rows());
                Eigen::MatrixXd P_ctrv = P_model.topLeftCorner(CTRV_DIM, CTRV_DIM);

                double v = (x_model.size() > 3) ? x_model(3) : 0.0;
                double yaw = (x_model.size() > 4) ? x_model(4) : 0.0;
                double c = std::cos(yaw);
                double s = std::sin(yaw);

                // J = d[x y z vx vy vz] / d[x y z v yaw yaw_rate]
                Eigen::Matrix<double, OUT, 6> J;
                J.setZero();
                J(0, 0) = 1.0;
                J(1, 1) = 1.0;
                J(2, 2) = 1.0;
                J(3, 3) = c;
                J(3, 4) = -v * s;
                J(4, 3) = s;
                J(4, 4) = v * c;

                Eigen::MatrixXd J_dyn = J.leftCols(CTRV_DIM);
                P_uni = J_dyn * P_ctrv * J_dyn.transpose();
                break;
            }
            default:
            {
                int dim = std::min<int>(P_model.rows(), OUT);
                P_uni.topLeftCorner(dim, dim) = P_model.topLeftCorner(dim, dim);
                break;
            }
        }

        P_uni = 0.5 * (P_uni + P_uni.transpose());
        return P_uni;
    }

    // ============================================================
    // Helper: unified covariance (6D) -> model covariance
    // unified_x = [x, y, z, vx, vy, vz]
    // ============================================================
    static constexpr int kUnifiedDim = 6;
    static Eigen::MatrixXd fromUnifiedCovariance(const Eigen::VectorXd& unified_x,
                                                 const Eigen::MatrixXd& P_uni,
                                                 int model_type,
                                                 const Eigen::MatrixXd& original_P)
    {
        Eigen::MatrixXd P_model = original_P;

        if (P_uni.rows() == 0 || P_uni.cols() == 0)
        {
            return P_model;
        }

        switch (model_type)
        {
            case rm_track::CV:
            case rm_track::CA:
            {
                // CV/CA 量纲一致，直接安全切块复制
                int dim = std::min<int>(P_model.rows(), kUnifiedDim);
                P_model.topLeftCorner(dim, dim) = P_uni.topLeftCorner(dim, dim);
                break;
            }
            case rm_track::CTRV:
            {
                // unified_x 是 [x, y, z, vx, vy, vz]
                double vx = unified_x(3);
                double vy = unified_x(4);
                
                // 增加极小值保护，防止除以 0 导致系统崩溃
                double v_sq = std::max(vx * vx + vy * vy, 1e-6); 
                double v = std::sqrt(v_sq);

                // 构建 6x6 的逆向雅可比矩阵 J_inv (从笛卡尔到极坐标的偏导数)
                Eigen::Matrix<double, 6, kUnifiedDim> J_inv;
                J_inv.setZero();
                J_inv(0, 0) = 1.0; // x
                J_inv(1, 1) = 1.0; // y
                J_inv(2, 2) = 1.0; // z

                // 速度和角度的偏导数
                J_inv(3, 3) = vx / v;         // dv / dvx
                J_inv(3, 4) = vy / v;         // dv / dvy
                J_inv(4, 3) = -vy / v_sq;     // dyaw / dvx
                J_inv(4, 4) = vx / v_sq;      // dyaw / dvy

                const int CTRV_DIM = std::min<int>(6, (int)P_model.rows());
                Eigen::MatrixXd J_dyn = J_inv.topRows(CTRV_DIM);
                
                // 核心转换公式：P_model = J_inv * P_uni * J_inv^T
                P_model.topLeftCorner(CTRV_DIM, CTRV_DIM) = J_dyn * P_uni * J_dyn.transpose();
                break;
            }
            default:
            {
                int dim = std::min<int>(P_model.rows(), kUnifiedDim);
                P_model.topLeftCorner(dim, dim) = P_uni.topLeftCorner(dim, dim);
                break;
            }
        }
        
        // 强制对称，防止 UKF 的 Cholesky 分解报错
        P_model = 0.5 * (P_model + P_model.transpose()); 
        return P_model;
    }
    // ============================================================
    // Map unified 6D → model-specific state
    // ============================================================
    Eigen::VectorXd AIMM::fromUnifiedState(const Eigen::VectorXd& unified, int model_type,
                                            const Eigen::VectorXd& original) const
    {
        Eigen::VectorXd state = original;  // preserve dimensions and extra states

        switch (model_type)
        {
            case rm_track::CV:  // [x,y,z,vx,vy,vz]
                state.head(OUTPUT_DIM) = unified;
                break;

            case rm_track::CA:  // [x,y,z,vx,vy,vz,ax,ay,az]
                state.head(OUTPUT_DIM) = unified;
                // keep acceleration from original
                break;

            case rm_track::CTRV:  // [x,y,z,v,yaw,yaw_rate]
            {
                state(0) = unified(0);  // x
                state(1) = unified(1);  // y
                state(2) = unified(2);  // z
                double vx = unified(3);
                double vy = unified(4);
                state(3) = std::sqrt(vx * vx + vy * vy);  // v
                state(4) = std::atan2(vy, vx);             // yaw
                // keep yaw_rate from original
                break;
            }
            default:
                for (int i = 0; i < std::min((int)unified.size(), (int)state.size()); ++i)
                    state(i) = unified(i);
                break;
        }
        return state;
    }

    // ============================================================
    // Step 0: Compute mixing probabilities μ_{i|j}
    // ============================================================
    void AIMM::computeMixingProbabilities()
    {
        // c_j = Σ_i π_{ij} * μ_i  (normalizing constants)
        Eigen::VectorXd c_bar = Eigen::VectorXd::Zero(NUM_MODELS);
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            for (int i = 0; i < NUM_MODELS; ++i)
            {
                c_bar(j) += TPM_(i, j) * mu_(i);
            }
            if (c_bar(j) < 1e-15) c_bar(j) = 1e-15;  // prevent division by zero
        }

        // μ_{i|j} = π_{ij} * μ_i / c_j
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            for (int i = 0; i < NUM_MODELS; ++i)
            {
                mixing_probs_(i, j) = TPM_(i, j) * mu_(i) / c_bar(j);
            }
        }
    }

    // ============================================================
    // Step 1: Interaction — mix states across models
    // ============================================================
    void AIMM::interactionStep()
    {
        computeMixingProbabilities();

        for (int j = 0; j < NUM_MODELS; ++j)
        {
            // Mixed mean in unified space
            Eigen::VectorXd x0j = Eigen::VectorXd::Zero(OUTPUT_DIM);
            for (int i = 0; i < NUM_MODELS; ++i)
            {
                Eigen::VectorXd xi_unified = toUnifiedState(filters_[i].getState(), model_types_[i]);
                x0j += mixing_probs_(i, j) * xi_unified;
            }

            // Mixed covariance in unified space
            Eigen::MatrixXd P0j = Eigen::MatrixXd::Zero(OUTPUT_DIM, OUTPUT_DIM);
            for (int i = 0; i < NUM_MODELS; ++i)
            {
                const Eigen::VectorXd xi_model = filters_[i].getState();
                const Eigen::MatrixXd Pi_model = filters_[i].getCovariance();

                Eigen::VectorXd xi_unified = toUnifiedState(xi_model, model_types_[i]);
                Eigen::VectorXd diff = xi_unified - x0j;

                // model -> unified
                Eigen::MatrixXd Pi_unified = toUnifiedCovariance(xi_model, Pi_model, model_types_[i]);

                P0j += mixing_probs_(i, j) * (Pi_unified + diff * diff.transpose());
            }

            P0j = 0.5 * (P0j + P0j.transpose());
            mixed_states_[j] = x0j;
            mixed_covs_[j] = P0j;
        }
    }

    // ============================================================
    // Step 4: Compute Gaussian likelihood for model j
    // ============================================================
    double AIMM::computeLikelihood(int model_idx, const Eigen::Vector3d& z_meas)
    {
        // Use UKF's sigma-point-based innovation computation for accurate S and z_pred
        // computeInnovation returns the NIS (Normalized Innovation Squared), which is exactly v^T * S^-1 * v
        Eigen::VectorXd z_pred;
        Eigen::MatrixXd S;
        double nis = filters_[model_idx].computeInnovation(z_meas, z_pred, S);

        // Gaussian likelihood: L = (2π)^{-d/2} |S|^{-1/2} exp(-0.5 * NIS)
        double det_S = S.determinant();
        if (det_S < 1e-30) det_S = 1e-30;

        int meas_dim = z_meas.size();

        // Directly use the returned NIS for the exponent, avoiding redundant S.inverse() computation
        double exponent = -0.5 * nis;
        
        // Save to internal state for debug topic later
        filters_[model_idx].last_likelihood_exp_ = exponent;

        // Clamp exponent to prevent underflow
        exponent = std::max(exponent, -500.0);

        // Using standard computation instead of manually doing 2pi^-d/2 / sqrt(|S|)
        double log_likelihood = -0.5 * (meas_dim * std::log(2.0 * M_PI) + std::log(det_S)) + exponent;
        double likelihood = std::exp(log_likelihood);

        filters_[model_idx].last_likelihood_ = std::max(likelihood, LIKELIHOOD_FLOOR);

        return filters_[model_idx].last_likelihood_;
    }

    // ============================================================
    // Step 5: Update model probabilities
    // ============================================================
    void AIMM::updateModelProbabilities(const Eigen::Vector3d& z_meas)
    {
        Eigen::VectorXd likelihoods(NUM_MODELS);
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            likelihoods(j) = computeLikelihood(j, z_meas);
        }

        // Predicted model probabilities: c_j = Σ_i π_{ij} * μ_i
        Eigen::VectorXd c_bar(NUM_MODELS);
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            c_bar(j) = 0;
            for (int i = 0; i < NUM_MODELS; ++i)
            {
                c_bar(j) += TPM_(i, j) * mu_(i);
            }
        }

        // μ_j = L_j * c_j / Σ_k(L_k * c_k)
        Eigen::VectorXd mu_new(NUM_MODELS);
        double total = 0;
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            mu_new(j) = likelihoods(j) * c_bar(j);
            total += mu_new(j);
        }

        if (total < 1e-300)
        {
            // All likelihoods near zero — reset to uniform
            mu_ = Eigen::VectorXd::Constant(NUM_MODELS, 1.0 / NUM_MODELS);
            ROS_WARN_THROTTLE(2, "[AIMM] All likelihoods near zero, resetting to uniform");
        }
        else
        {
            mu_ = mu_new / total;
        }

        ROS_DEBUG_THROTTLE(1, "[AIMM] Probabilities: CV=%.3f CA=%.3f CTRV=%.3f",
                        mu_(0), mu_(1), mu_(2));
    }

    // ============================================================
    // Step 6: Combine estimates (probability-weighted)
    // ============================================================
    void AIMM::combineEstimates()
    {
        // Combined state in unified space
        combined_state_ = Eigen::VectorXd::Zero(OUTPUT_DIM);
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            Eigen::VectorXd xj_unified = toUnifiedState(filters_[j].getState(), model_types_[j]);
            combined_state_ += mu_(j) * xj_unified;
        }

        // Combined covariance in unified space
        combined_cov_ = Eigen::MatrixXd::Zero(OUTPUT_DIM, OUTPUT_DIM);
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            const Eigen::VectorXd xj_model = filters_[j].getState();
            const Eigen::MatrixXd Pj_model = filters_[j].getCovariance();

            Eigen::VectorXd xj_unified = toUnifiedState(xj_model, model_types_[j]);
            Eigen::VectorXd diff = xj_unified - combined_state_;

            // model -> unified
            Eigen::MatrixXd Pj_unified = toUnifiedCovariance(xj_model, Pj_model, model_types_[j]);

            combined_cov_ += mu_(j) * (Pj_unified + diff * diff.transpose());
        }
        combined_cov_ = 0.5 * (combined_cov_ + combined_cov_.transpose());
    }

    // ============================================================
    // ADAPTIVE: Compute per-model NIS (Normalised Innovation Squared)
    // NIS_j = ν_j^T S_j^{-1} ν_j   where ν = z - z_pred, S computed via sigma points
    // Uses UKF's sigma-point-based computation for accurate S
    // ============================================================
    double AIMM::computeNIS(int model_idx, const Eigen::Vector3d& z_meas)
    {
        Eigen::VectorXd z_pred;
        Eigen::MatrixXd S;
        double nis = filters_[model_idx].computeInnovation(z_meas, z_pred, S);

        // Optional log for debugging NIS, can only retrieve innovation here for logging
        // We only compute innovation to print it out
        Eigen::Vector3d innovation = z_meas - z_pred.head(3);

        filters_[model_idx].last_innovation_ = innovation;
        filters_[model_idx].last_S_ = S;
        filters_[model_idx].last_nis_ = nis;

        ROS_INFO_THROTTLE(2, "[AIMM::NIS] model=%s z=(%.4f,%.4f,%.4f) pred=(%.4f,%.4f,%.4f) "
                          "innov=(%.4f,%.4f,%.4f) S_diag=(%.4f,%.4f,%.4f) NIS=%.6f",
                          model_names_[model_idx].c_str(),
                          z_meas(0), z_meas(1), z_meas(2),
                          z_pred(0), z_pred(1), z_pred(2),
                          innovation(0), innovation(1), innovation(2),
                          S(0,0), S(1,1), S(2,2), nis);

        return nis;
    }

    // ============================================================
    // ADAPTIVE: Update maneuver indicator λ(k) after measurement
    // 
    // Uses a self-calibrating approach:
    //   1. Compute raw NIS (probability-weighted across models)
    //   2. Maintain a slow-adapting baseline NIS (the "expected" NIS when
    //      the filter is well-matched). This handles the case where Q is
    //      large and raw NIS is always much smaller than the theoretical 3.0.
    //   3. λ = NIS_raw / NIS_baseline (ratio > 1 means maneuvering)
    //   4. Apply EMA smoothing to λ
    //
    // Under correct model: NIS_raw ≈ NIS_baseline → λ ≈ 1.0
    // Under maneuver:      NIS_raw >> NIS_baseline → λ >> 1.0
    // ============================================================
    void AIMM::updateManeuverIndicator(const Eigen::Vector3d& z_meas)
    {
        // Compute per-model NIS (using sigma-point-based innovation from UKF)
        double nis_combined = 0.0;
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            model_NIS_[j] = computeNIS(j, z_meas);
            nis_combined += mu_(j) * model_NIS_[j];
        }

        // 首帧直接将 baseline 初始化为当前 NIS 值
        // 避免从错误的初始值（如 0.1）慢慢收敛，导致 λ 长时间异常
        if (!nis_baseline_initialized_)
        {
            nis_baseline_ = std::max(nis_combined, 1e-6);
            nis_baseline_initialized_ = true;
            lambda_ = LAMBDA_REF;  // 首帧 λ = 1.0（无机动假设）
            ROS_INFO("[AIMM] NIS baseline initialized to %.6f (first frame)", nis_baseline_);
            return;
        }

        // 自适应 baseline：用慢速 EMA 追踪"正常"NIS 水平
        // 但只在 λ 接近 nominal 时更新（机动时不更新，防止 baseline 被拉高）
        if (lambda_ < LAMBDA_REF * 2.0)
        {
            nis_baseline_ = nis_baseline_alpha_ * nis_combined 
                           + (1.0 - nis_baseline_alpha_) * nis_baseline_;
        }
        // Floor to prevent division by zero
        nis_baseline_ = std::max(nis_baseline_, 1e-8);

        // λ = ratio of current NIS to baseline
        // λ ≈ 1.0 when filter is well-matched (no maneuver)
        // λ >> 1.0 when NIS spikes (maneuver detected)
        double lambda_raw = nis_combined / nis_baseline_;

        // Exponential moving average to smooth jitter
        lambda_ = lambda_ema_alpha_ * lambda_raw + (1.0 - lambda_ema_alpha_) * lambda_;
        
        // Clamp to prevent runaway
        lambda_ = std::min(lambda_, 100.0);

        ROS_INFO_THROTTLE(2, "[AIMM::lambda] NIS_per_model=(%.6f,%.6f,%.6f) mu=(%.3f,%.3f,%.3f) "
                          "nis_raw=%.6f nis_baseline=%.6f lambda_ratio=%.4f lambda=%.4f",
                          model_NIS_[0], model_NIS_[1], model_NIS_[2],
                          mu_(0), mu_(1), mu_(2),
                          nis_combined, nis_baseline_, lambda_raw, lambda_);
    }

    // ============================================================
    // ADAPTIVE: Adjust TPM based on maneuver indicator λ
    //
    // Core idea:
    //   λ small (≤ LAMBDA_REF) → target is smooth → high p_stay
    //   λ large (>> LAMBDA_REF) → target is maneuvering → low p_stay
    // ============================================================
    void AIMM::adaptTPM(const Eigen::Vector3d& innovation, const Eigen::Vector3d& vel)
    {
        // Sigmoid-based maneuver level ∈ [0, 1]
        // 0 = no maneuver, 1 = extreme maneuver
        double ratio = lambda_ / LAMBDA_REF;  // 1.0 = nominal
        double maneuver_level = 1.0 / (1.0 + std::exp(-sigmoid_k_ * (ratio - 2.0)));
        
        // Interpolate p_stay
        double p_stay = p_stay_max_ - (p_stay_max_ - p_stay_min_) * maneuver_level; 
        double p_switch_total = 1.0 - p_stay;
        double alpha =0.5;
        double v_norm = vel.norm();
        double inno_num = innovation.norm();

        if(v_norm>1e-3 && inno_num>1e-3) // only compute alpha when both velocity and innovation are significant to avoid noise
        {
            Eigen::Vector3d v_dir = vel.normalized(); // get direction of velocity
            double e_tan_mag = std::abs(innovation.dot(v_dir)); // dot product gives magnitude of innovation in direction of motion
            double e_normal_sq = std::max(0.0,innovation.squaredNorm() - e_tan_mag * e_tan_mag); // Pythagorean theorem to get magnitude of innovation perpendicular to motion
            double e_normal_mag = std::sqrt(e_normal_sq);

            alpha = e_normal_mag / inno_num; // 0 = purely tangential (straight), 1 = purely normal (turning)
        }


        // double off_weight[NUM_MODELS][NUM_MODELS];
        
        // // Weight bias: how much to favor high-maneuver models when λ is large
        // double ca_bias = 1.0 + 2.0 * maneuver_level;    // CA gets more weight when maneuvering
        // double ctrv_bias = 1.0 + 1.5 * maneuver_level;  // CTRV gets some weight
        // double cv_bias = 1.0 + 2.0 * (1.0 - maneuver_level); // CV gets weight when smooth
        
        // // From CV (index 0): switch to CA or CTRV
        // off_weight[0][0] = 0;
        // off_weight[0][1] = ca_bias;    // CV → CA
        // off_weight[0][2] = ctrv_bias;  // CV → CTRV
        
        // // From CA (index 1): switch to CV or CTRV
        // off_weight[1][0] = cv_bias;    // CA → CV
        // off_weight[1][1] = 0;
        // off_weight[1][2] = ctrv_bias;  // CA → CTRV
        
        // // From CTRV (index 2): switch to CV or CA
        // off_weight[2][0] = cv_bias;    // CTRV → CV
        // off_weight[2][1] = ca_bias;    // CTRV → CA
        // off_weight[2][2] = 0;

        // // Normalize off-diagonal per row and fill TPM
        // for (int i = 0; i < NUM_MODELS; ++i)
        // {
        //     double row_sum = 0;
        //     for (int j = 0; j < NUM_MODELS; ++j)
        //     {
        //         if (i != j) row_sum += off_weight[i][j];
        //     }
            
        //     for (int j = 0; j < NUM_MODELS; ++j)
        //     {
        //         if (i == j)
        //             TPM_(i, j) = p_stay;
        //         else
        //             TPM_(i, j) = p_switch_total * off_weight[i][j] / std::max(row_sum, 1e-10);
        //     }
        // }


        //TODO : 提取 2.0--maneuver_gain 权重系数 加入动态调参
        // bias
        double cv_bias = 1.0 + maneuver_gain_ * (1.0 - maneuver_level);   // CV gets more weight when motion is straight
        double ca_bias = 1.0 + maneuver_gain_ * maneuver_level * (1.0 - alpha);   // CA gets more weight when motion is accelerating (tangential innovation)
        double ctrv_bias = 1.0 + maneuver_gain_ * maneuver_level * alpha;  // CTRV gets some weight when maneuvering, regardless of innovation direction

        Eigen::Matrix3d bias_matrix;
        bias_matrix << 0, ca_bias, ctrv_bias,
                       cv_bias, 0, ctrv_bias,
                       cv_bias, ca_bias, 0;
        
        // normalized bias matrix
        Eigen::Vector3d row_sums = bias_matrix.rowwise().sum().cwiseMax(Eigen::Vector3d::Constant(1e-10)); // add up as rows, and prevent division by zero
        Eigen::Matrix3d normalized_bias = bias_matrix.array().colwise() / row_sums.array(); // normalize each row to sum to 1


        TPM_ = normalized_bias * p_switch_total;
        TPM_.diagonal() = Eigen::Vector3d::Constant(p_stay);


       // TODO 修改debug 逻辑
        // ROS_DEBUG_THROTTLE(1, "[AIMM] adaptTPM: lambda=%.2f, level=%.2f, p_stay=%.3f", 
        //                    lambda_, maneuver_level, p_stay);
    }

    // ============================================================
    // ADAPTIVE: Scale each model's Q matrix based on maneuver indicator λ
    //
    // α(k) = clamp(λ / λ_ref, 1.0, q_scale_max_)
    //
    // Q_j(k) = α(k) * Q_base_j
    //
    // This widens the process noise during maneuvers so the filter
    // trusts its prediction less and tracks measurements more closely.
    // During smooth motion α → 1.0, so Q returns to its base value
    // set by dynamic_reconfigure.
    // ============================================================
    void AIMM::adaptProcessNoise()
    {
        double alpha = std::max(1.0, lambda_ / LAMBDA_REF);
        alpha = std::min(alpha, q_scale_max_);
        
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            // Always read the *config-derived* base Q from UKF.
            // This is the Q set by dynamic_reconfigure (ukfconfigCB),
            // NOT the scaled Q we wrote via setProcessNoise last frame.
            // This avoids the feedback loop where scaled Q becomes the new base.
            Eigen::MatrixXd Q_cfg = filters_[j].getBaseProcessNoise();
            if (Q_cfg.size() > 0)
            {
                Q_base_[j] = Q_cfg;
            }
            
            Eigen::MatrixXd Q_adapted = Q_base_[j] * alpha;
            filters_[j].setProcessNoise(Q_adapted);
        }
        
        ROS_INFO_THROTTLE(2, "[AIMM] adaptQ: lambda=%.4f, alpha=%.2f, Q_base[0]_pos=%.6f, Q_actual[0]_pos=%.6f, P[0]_pos=%.4f",
                          lambda_, alpha,
                          Q_base_[0].rows() > 0 ? Q_base_[0](0,0) : -1.0,
                          filters_[0].getProcessNoise()(0,0),
                          filters_[0].getCovariance()(0,0));
    }

    // ============================================================
    // Main pipeline
    // ============================================================
    void AIMM::predict()
    {
        if (!initialized_) return;

        // Step 0a: Adapt TPM based on maneuver indicator from previous step
        // Use innovation and velocity from the most probable model

        adaptTPM(last_innovation_, last_velocity_);
        
        // Step 0b: Adapt process noise Q for each model
        adaptProcessNoise();

        // Step 1: Interaction (mix states across models)
        interactionStep();

        // Inject mixed states back into each filter
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            Eigen::VectorXd current_state = filters_[j].getState();
            Eigen::MatrixXd current_P = filters_[j].getCovariance();
            Eigen::VectorXd mixed_state_model = fromUnifiedState(mixed_states_[j], model_types_[j], current_state);
            Eigen::MatrixXd mixed_cov_model = fromUnifiedCovariance(mixed_states_[j], mixed_covs_[j], model_types_[j], current_P);

            filters_[j].setState(mixed_state_model, mixed_cov_model);
        }

        // Step 2: Each model predicts independently
        for (auto& f : filters_)
        {
            f.predict();
        }
    }

    void AIMM::update(const Eigen::Vector3d& z_meas)
    {
        if (!initialized_) return;

        // ============================================================
        // CRITICAL: 必须在 UKF update 之前计算 NIS / 似然度 / λ
        // 因为 update 之后 state 被量测修正，innovation ≈ 0，λ 永远为 0
        // 
        // 正确顺序（此时各模型 state = predict 之后的纯预测值）:
        //   Step 3a: 计算 NIS 和似然度（用预测状态 vs 量测）
        //   Step 3b: 更新模型概率
        //   Step 3c: 更新机动指标 λ
        //   Step 4:  各模型执行 UKF update（状态被量测修正）
        //   Step 5:  组合输出
        // ============================================================

        // Step 3a-3b: Update model probabilities (uses predicted state, BEFORE update)
        updateModelProbabilities(z_meas);

        // Step 3c: Update maneuver indicator λ (uses predicted state, BEFORE update)
        updateManeuverIndicator(z_meas);

        // save cv model innovation as reference for next frame's interaction step
        Eigen::VectorXd z_pred_cv;
        Eigen::MatrixXd S_cv;
        filters_[0].computeInnovation(z_meas, z_pred_cv, S_cv);
        last_innovation_ = z_meas - z_pred_cv.head<3>();



        // Step 4: NOW do the actual UKF update for each model
        for (auto& f : filters_)
        {
            f.update(z_meas);
        }

        // Step 5: Combine estimates (uses post-update states)
        combineEstimates();

        //save combined velocity as reference for next frame's interaction step
        last_velocity_ = combined_state_.segment<3>(3);

        ROS_INFO_THROTTLE(2, "[AIMM] Active: %s (%.1f%%), lambda=%.6f, state=(%.3f,%.3f,%.3f)",
                        getActiveModelName().c_str(),
                        mu_(getActiveModelIndex()) * 100.0,
                        lambda_,
                        combined_state_(0), combined_state_(1), combined_state_(2));
    }

    Eigen::VectorXd AIMM::getState() const
    {
        return combined_state_;
    }

    Eigen::MatrixXd AIMM::getCovariance() const
    {
        return combined_cov_;
    }

    int AIMM::getActiveModelIndex() const
    {
        int best = 0;
        for (int i = 1; i < NUM_MODELS; ++i)
        {
            if (mu_(i) > mu_(best)) best = i;
        }
        return best;
    }

    std::string AIMM::getActiveModelName() const
    {
        return model_names_[getActiveModelIndex()];
    }

    int AIMM::getModelType() const
    {
        return model_types_[getActiveModelIndex()];
    }

    rm_radar_msgs::aimm_debugger AIMM::getDebugMsg() const
    {
        rm_radar_msgs::aimm_debugger msg;

        if (mu_.size() >= 3)
        {
            msg.mu_cv = mu_(0);
            msg.mu_ca = mu_(1);
            msg.mu_ctrv = mu_(2);
        }

        msg.lambda = lambda_;
        
        // Compute overall NIS raw for debugging info
        double nis_combined = 0.0;
        for (int j = 0; j < NUM_MODELS; ++j) {
            nis_combined += mu_(j) * model_NIS_[j];
        }
        msg.nis_raw = nis_combined;
        msg.nis_baseline = nis_baseline_;
        
        double alpha = std::max(1.0, lambda_ / LAMBDA_REF);
        msg.q_scale_alpha = std::min(alpha, q_scale_max_);
        msg.active_model = getActiveModelIndex();

        std::vector<rm_radar_msgs::ukf_debugger*> ukf_msgs = {&msg.cv_debug, &msg.ca_debug, &msg.ctrv_debug};

        for (int i = 0; i < NUM_MODELS; ++i)
        {
            ukf_msgs[i]->nis = filters_[i].last_nis_;
            ukf_msgs[i]->likelihood = filters_[i].last_likelihood_;
            ukf_msgs[i]->likelihood_exp = filters_[i].last_likelihood_exp_;
            
            // Vector conversions
            Eigen::VectorXd state = filters_[i].getState();
            ukf_msgs[i]->state.resize(state.size());
            for(int k=0; k<state.size(); ++k) ukf_msgs[i]->state[k] = state(k);

            Eigen::VectorXd innov = filters_[i].last_innovation_;
            ukf_msgs[i]->innovation.resize(innov.size());
            for(int k=0; k<innov.size(); ++k) ukf_msgs[i]->innovation[k] = innov(k);

            Eigen::MatrixXd P = filters_[i].getCovariance();
            ukf_msgs[i]->p_diag.resize(P.rows());
            for(int k=0; k<P.rows(); ++k) ukf_msgs[i]->p_diag[k] = P(k,k);

            Eigen::MatrixXd S = filters_[i].last_S_;
            ukf_msgs[i]->s_diag.resize(S.rows());
            for(int k=0; k<S.rows(); ++k) ukf_msgs[i]->s_diag[k] = S(k,k);

            Eigen::MatrixXd Q = filters_[i].getProcessNoise();
            ukf_msgs[i]->q_diag.resize(Q.rows());
            for(int k=0; k<Q.rows(); ++k) ukf_msgs[i]->q_diag[k] = Q(k,k);
        }

        return msg;
    }

}  // namespace rm_radarplugin
