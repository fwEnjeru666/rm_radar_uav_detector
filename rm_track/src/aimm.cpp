#include "aimm.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace rm_radarplugin
{

constexpr double AIMM::LIKELIHOOD_FLOOR;
    AIMM::AIMM(ros::NodeHandle& nh, const std::vector<std::shared_ptr<BaseModel>>& models)
        : nh_(nh)
    {

        // Define model types
        num_models_ = models.size();

        if(num_models_ < 2)
        {
            ROS_ERROR("[AIMM] Error: At least 2 models are required for AIMM. Provided: %d", num_models_);
            throw std::runtime_error("AIMM requires at least 2 models");
        }

        model_names_.clear();
        for(const auto& model : models)
        {
            model_names_.push_back(model->getName());
        }

        // Create one UKF per model, each with a unique sub-namespace under "aimm"
        filters_.clear();
        filters_.reserve(num_models_);
        for(int i = 0; i < num_models_; i++) 
        {
            ros::NodeHandle model_nh(nh_, "aimm/" + model_names_[i]);
            filters_.emplace_back(std::make_shared<UKF>(model_nh, models[i]));
        }

        // Initial model probabilities (uniform)
        mu_ = Eigen::VectorXd::Constant(num_models_, 1.0 / num_models_);

        // Initial TPM — will be overwritten every step by adaptTPM()
        TPM_ = Eigen::MatrixXd::Zero(num_models_, num_models_);
        double p_stay = p_stay_nominal_;
        double p_switch = (1.0 - p_stay) / (num_models_ - 1);
        for (int i = 0; i < num_models_; ++i)
        {
            for (int j = 0; j < num_models_; ++j)
            {
                TPM_(i, j) = (i == j) ? p_stay : p_switch;
            }
        }
        
        // Allocate mixing storage
        mixed_states_.resize(num_models_);
        mixed_covs_.resize(num_models_);
        mixing_probs_ = Eigen::MatrixXd::Zero(num_models_, num_models_);

        // Adaptive storage
        model_NIS_.resize(num_models_, 0.0);
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

        std::ostringstream model_list;
        for (size_t i = 0; i < model_names_.size(); ++i)
        {
            if (i > 0) model_list << ", ";
            model_list << model_names_[i];
        }
        ROS_INFO("[AIMM] Created with %d models: %s", num_models_, model_list.str().c_str());
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

    void AIMM::initialize(const Eigen::Vector3d& z_meas)
    {
        mu_ = Eigen::VectorXd::Constant(num_models_, 1.0 / num_models_);  // Reset model probabilities
        lambda_ = lambda_ref_;  // Reset maneuver intensity
        nis_baseline_ = 0.0;
        nis_baseline_initialized_ = false;

        double p_stay = p_stay_nominal_;
        double p_switch = (1.0 - p_stay) / (std::max(1, num_models_ -1));
        
        last_innovation_ = Eigen::Vector3d::Zero();
        last_velocity_   = Eigen::Vector3d::Zero();

        for(int i = 0; i<num_models_; ++i)
        {
            filters_[i]->initialize(z_meas);

            model_NIS_[i] = 0.0;  // Reset NIS for all models

            for(int j = 0; j<num_models_; j++)
            {
                TPM_(i,j) = (i==j) ? p_stay : p_switch;
            }
        }

        combined_state_ = Eigen::VectorXd::Zero(OUTPUT_DIM);
        combined_state_.head(3) = z_meas;
        combined_cov_ = Eigen::MatrixXd::Identity(OUTPUT_DIM, OUTPUT_DIM) * 0.1;

        initialized_ = true;
        ROS_INFO("[AIMM] Initialized with measurement: [%.2f, %.2f, %.2f]", z_meas(0), z_meas(1), z_meas(2));


    }

    void AIMM::setDt(double dt)
    {
        for (auto& f : filters_)
        {
            f->setDt(dt);
        }
    }

    void AIMM::setMeasurementNoise(const Eigen::MatrixXd& R)
    {
        for (auto& f : filters_)
        {
            f->setMeasurementNoise(R);
        }
    }

    Eigen::MatrixXd AIMM::getMeasurementNoise() const
    {
        if (!filters_.empty())
            return filters_[0]->getMeasurementNoise();
        return Eigen::Matrix3d::Identity() * 0.1;
    }

    Eigen::MatrixXd AIMM::getBaseMeasurementNoise() const
    {
        if (!filters_.empty())
            return filters_[0]->getBaseMeasurementNoise();
        return Eigen::Matrix3d::Identity() * 0.1;
    }

    // ============================================================
    // Step 0: Compute mixing probabilities μ_{i|j}
    // ============================================================
    void AIMM::computeMixingProbabilities()
    {
        // c_j = Σ_i π_{ij} * μ_i  (normalizing constants)
        Eigen::VectorXd c_bar = Eigen::VectorXd::Zero(num_models_);
        for (int j = 0; j < num_models_; ++j)
        {
            for (int i = 0; i < num_models_; ++i)
            {
                c_bar(j) += TPM_(i, j) * mu_(i);
            }
            if (c_bar(j) < 1e-15) c_bar(j) = 1e-15;  // prevent division by zero
        }

        // μ_{i|j} = π_{ij} * μ_i / c_j
        for (int j = 0; j < num_models_; ++j)
        {
            for (int i = 0; i < num_models_; ++i)
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

        for (int j = 0; j < num_models_; ++j)
        {
            // Mixed mean in unified space
            Eigen::VectorXd x0j = Eigen::VectorXd::Zero(OUTPUT_DIM);
            for (int i = 0; i < num_models_; ++i)
            {
                ModelState current = {filters_[i]->getState(), filters_[i]->getCovariance()};
                ModelState unified = filters_[i]->getModel()->toUnifiedState(current);
                x0j += mixing_probs_(i, j) * unified.state;
            }

            // Mixed covariance in unified space
            Eigen::MatrixXd P0j = Eigen::MatrixXd::Zero(OUTPUT_DIM, OUTPUT_DIM);
            for (int i = 0; i < num_models_; ++i)
            {
                ModelState current = {filters_[i]->getState(), filters_[i]->getCovariance()};
                ModelState unified = filters_[i]->getModel()->toUnifiedState(current);
                Eigen::VectorXd diff = unified.state.head(OUTPUT_DIM) - x0j;
                P0j += mixing_probs_(i, j) * (unified.covariance.topLeftCorner(OUTPUT_DIM, OUTPUT_DIM) + diff * diff.transpose());
            }

            P0j = 0.5 * (P0j + P0j.transpose());
            mixed_states_[j] = x0j;
            mixed_covs_[j] = P0j;
        }
    }

    // ============================================================
    // Step 4: Compute Gaussian likelihood for model j
    // ============================================================
    double AIMM::computeLikelihood(int model_idx, const Eigen::VectorXd& z_meas)
    {
        // Use UKF's sigma-point-based innovation computation for accurate S and z_pred
        // computeInnovation returns the NIS (Normalized Innovation Squared), which is exactly v^T * S^-1 * v
        Eigen::VectorXd z_pred;
        Eigen::MatrixXd S;
        double nis = filters_[model_idx]->computeInnovation(z_meas, z_pred, S);

        // Gaussian likelihood: L = (2π)^{-d/2} |S|^{-1/2} exp(-0.5 * NIS)
        double det_S = S.determinant();
        if (det_S < 1e-30) det_S = 1e-30;

        int meas_dim = z_meas.size();

        // Directly use the returned NIS for the exponent, avoiding redundant S.inverse() computation
        double exponent = -0.5 * nis;
        
        // Clamp exponent to prevent underflow
        exponent = std::max(exponent, -500.0);

        // Using standard computation instead of manually doing 2pi^-d/2 / sqrt(|S|)
        double log_likelihood = -0.5 * (meas_dim * std::log(2.0 * M_PI) + std::log(det_S)) + exponent;
        double likelihood = std::exp(log_likelihood);

        return likelihood;
    }

    // ============================================================
    // Step 5: Update model probabilities
    // ============================================================
    void AIMM::updateModelProbabilities(const Eigen::VectorXd& z_meas)
    {
        Eigen::VectorXd likelihoods(num_models_);
        for (int j = 0; j < num_models_; ++j)
        {
            likelihoods(j) = computeLikelihood(j, z_meas);
        }

        // Predicted model probabilities: c_j = Σ_i π_{ij} * μ_i
        Eigen::VectorXd c_bar(num_models_);
        for (int j = 0; j < num_models_; ++j)
        {
            c_bar(j) = 0;
            for (int i = 0; i < num_models_; ++i)
            {
                c_bar(j) += TPM_(i, j) * mu_(i);
            }
        }

        // μ_j = L_j * c_j / Σ_k(L_k * c_k)
        Eigen::VectorXd mu_new(num_models_);
        double total = 0;
        for (int j = 0; j < num_models_; ++j)
        {
            mu_new(j) = likelihoods(j) * c_bar(j);
            total += mu_new(j);
        }

        if (total < 1e-300)
        {
            // All likelihoods near zero — reset to uniform
            mu_ = Eigen::VectorXd::Constant(num_models_, 1.0 / num_models_);
            ROS_WARN_THROTTLE(2, "[AIMM] All likelihoods near zero, resetting to uniform");
        }
        else
        {
            mu_ = mu_new / total;
        }

        std::ostringstream prob_ss;
        prob_ss << "[AIMM] Probabilities:";
        for (int i = 0; i < num_models_; ++i)
        {
            const std::string& name = (i < static_cast<int>(model_names_.size())) ? model_names_[i] : std::string("model") + std::to_string(i);
            prob_ss << " " << name << "=" << mu_(i);
        }
        ROS_DEBUG_THROTTLE(1, "%s", prob_ss.str().c_str());
    }

    // ============================================================
    // Step 6: Combine estimates (probability-weighted)
    // ============================================================
    void AIMM::combineEstimates()
    {
        // Combined state in unified space
        combined_state_ = Eigen::VectorXd::Zero(OUTPUT_DIM);
        for (int j = 0; j < num_models_; ++j)
        {
            ModelState current = {filters_[j]->getState(), filters_[j]->getCovariance()};
            ModelState unified = filters_[j]->getModel()->toUnifiedState(current);
            combined_state_ += mu_(j) * unified.state;
        }

        // Combined covariance in unified space
        combined_cov_ = Eigen::MatrixXd::Zero(OUTPUT_DIM, OUTPUT_DIM);
        for (int j = 0; j < num_models_; ++j)
        {
            ModelState current = {filters_[j]->getState(), filters_[j]->getCovariance()};
            ModelState unified = filters_[j]->getModel()->toUnifiedState(current);
            Eigen::VectorXd diff = unified.state.head(OUTPUT_DIM) - combined_state_;
            combined_cov_ += mu_(j) * (unified.covariance.topLeftCorner(OUTPUT_DIM, OUTPUT_DIM) + diff * diff.transpose());
        }
        combined_cov_ = 0.5 * (combined_cov_ + combined_cov_.transpose());
    }

    // ============================================================
    // ADAPTIVE: Compute per-model NIS (Normalised Innovation Squared)
    // NIS_j = ν_j^T S_j^{-1} ν_j   where ν = z - z_pred, S computed via sigma points
    // Uses UKF's sigma-point-based computation for accurate S
    // ============================================================
    double AIMM::computeNIS(int model_idx, const Eigen::VectorXd& z_meas)
    {
        Eigen::VectorXd z_pred;
        Eigen::MatrixXd S;
        double nis = filters_[model_idx]->computeInnovation(z_meas, z_pred, S);

        if (z_meas.size() >= 3 && z_pred.size() >= 3 && S.rows() >= 3 && S.cols() >= 3)
        {
            Eigen::Vector3d innovation = z_meas.head(3) - z_pred.head(3);
            ROS_INFO_THROTTLE(2, "[AIMM::NIS] model=%s z=(%.4f,%.4f,%.4f) pred=(%.4f,%.4f,%.4f) "
                              "innov=(%.4f,%.4f,%.4f) S_diag=(%.4f,%.4f,%.4f) NIS=%.6f",
                              model_names_[model_idx].c_str(),
                              z_meas(0), z_meas(1), z_meas(2),
                              z_pred(0), z_pred(1), z_pred(2),
                              innovation(0), innovation(1), innovation(2),
                              S(0,0), S(1,1), S(2,2), nis);
        }

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
    void AIMM::updateManeuverIndicator(const Eigen::VectorXd& z_meas)
    {
        // Compute per-model NIS (using sigma-point-based innovation from UKF)
        double nis_combined = 0.0;
        for (int j = 0; j < num_models_; ++j)
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
            lambda_ = lambda_ref_;  // 首帧 λ = 1.0（无机动假设）
            ROS_INFO("[AIMM] NIS baseline initialized to %.6f (first frame)", nis_baseline_);
            return;
        }

        // 自适应 baseline：用慢速 EMA 追踪"正常"NIS 水平
        // 但只在 λ 接近 nominal 时更新（机动时不更新，防止 baseline 被拉高）
        if (lambda_ < lambda_ref_ * 2.0)
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

        std::ostringstream nis_ss;
        nis_ss << "[AIMM::lambda] ";
        for (int i = 0; i < num_models_; ++i)
        {
            const std::string& name = (i < static_cast<int>(model_names_.size())) ? model_names_[i] : std::string("model") + std::to_string(i);
            nis_ss << name << "(NIS=" << model_NIS_[i] << ",mu=" << mu_(i) << ") ";
        }
        nis_ss << "nis_raw=" << nis_combined
               << " nis_baseline=" << nis_baseline_
               << " lambda_ratio=" << lambda_raw
               << " lambda=" << lambda_;
        ROS_INFO_THROTTLE(2, "%s", nis_ss.str().c_str());
    }

    // ============================================================
    // ADAPTIVE: Adjust TPM based on maneuver indicator λ
    //
    // Core idea:
    //   λ small (≤ lambda_ref_) → target is smooth → high p_stay
    //   λ large (>> lambda_ref_) → target is maneuvering → low p_stay
    // ============================================================
    void AIMM::adaptTPM(const Eigen::Vector3d& innovation, const Eigen::Vector3d& vel)
    {
        // Sigmoid-based maneuver level ∈ [0, 1]
        // 0 = no maneuver, 1 = extreme maneuver
        double ratio = lambda_ / lambda_ref_;  // 1.0 = nominal
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


        TPM_ = Eigen::MatrixXd::Zero(num_models_, num_models_);

        if (num_models_ == 3)
        {
            // Keep the original 3-model bias behavior for CV/CA/CTRV.
            double cv_bias = 1.0 + maneuver_gain_ * (1.0 - maneuver_level);
            double ca_bias = 1.0 + maneuver_gain_ * maneuver_level * (1.0 - alpha);
            double ctrv_bias = 1.0 + maneuver_gain_ * maneuver_level * alpha;

            Eigen::Matrix3d bias_matrix;
            bias_matrix << 0, ca_bias, ctrv_bias,
                           cv_bias, 0, ctrv_bias,
                           cv_bias, ca_bias, 0;

            Eigen::Vector3d row_sums = bias_matrix.rowwise().sum().cwiseMax(Eigen::Vector3d::Constant(1e-10));
            Eigen::Matrix3d normalized_bias = bias_matrix.array().colwise() / row_sums.array();

            TPM_ = normalized_bias * p_switch_total;
            TPM_.diagonal() = Eigen::Vector3d::Constant(p_stay);
        }
        else
        {
            // For arbitrary model count, distribute switching probability uniformly.
            // This avoids hardcoded 3-model assumptions and dimension mismatch.
            double p_switch = (num_models_ > 1) ? (p_switch_total / static_cast<double>(num_models_ - 1)) : 0.0;
            for (int i = 0; i < num_models_; ++i)
            {
                for (int j = 0; j < num_models_; ++j)
                {
                    TPM_(i, j) = (i == j) ? p_stay : p_switch;
                }
            }
        }


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
        
        for (int j = 0; j < num_models_; ++j)
        {
            filters_[j]->setQScale(alpha);
        }
        
        ROS_INFO_THROTTLE(2, "[AIMM] adaptQ: lambda=%.4f, alpha=%.2f", lambda_, alpha);
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
        for (int j = 0; j < num_models_; ++j)
        {
            ModelState current = {filters_[j]->getState(), filters_[j]->getCovariance()};
            
            // get correct combined state dimension from unified state
            ModelState unified = filters_[j]->getModel()->toUnifiedState(current);
            int unified_dim = unified.state.size();
            
            ModelState uni_input;
            uni_input.state = Eigen::VectorXd::Zero(unified_dim);
            uni_input.state.head(OUTPUT_DIM) = mixed_states_[j];  // only overwrite position and velocity part
            uni_input.covariance = Eigen::MatrixXd::Identity(unified_dim, unified_dim);
            uni_input.covariance.topLeftCorner(OUTPUT_DIM, OUTPUT_DIM) = mixed_covs_[j]; // only overwrite position and velocity covariance

            ModelState restored = filters_[j]->getModel()->fromUnifiedState(uni_input);
            filters_[j]->setState(restored.state, restored.covariance);
            filters_[j]->predict();
        }
    }

    void AIMM::update(const Eigen::VectorXd& z_meas)
    {
        if (!initialized_) return;
        // Step 3a-3b: Update model probabilities (uses predicted state, BEFORE update)
        updateModelProbabilities(z_meas);

        // Step 3c: Update maneuver indicator λ (uses predicted state, BEFORE update)
        updateManeuverIndicator(z_meas);

        // save cv model innovation as reference for next frame's interaction step
        Eigen::VectorXd z_pred_cv;
        Eigen::MatrixXd S_cv;
        if(num_models_ > 0)
        {
            filters_[0]->computeInnovation(z_meas, z_pred_cv, S_cv);
            if (z_meas.size() >= 3 && z_pred_cv.size() >= 3)
            {
                last_innovation_ = z_meas.head(3) - z_pred_cv.head(3);
            }
            else
            {
                last_innovation_.setZero();
            }
        }

        // Step 4: NOW do the actual UKF update for each model
        for (auto& f : filters_)
        {
            f->update(z_meas);
        }

        // Step 5: Combine estimates (uses post-update states)
        combineEstimates();

        //save combined velocity as reference for next frame's interaction step
        last_velocity_ = combined_state_.segment<3>(3);
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
        for (int i = 1; i < num_models_; ++i)
        {
            if (mu_(i) > mu_(best)) best = i;
        }
        return best;
    }
    
    std::string AIMM::getActiveModelName() const
    {
        return model_names_[getActiveModelIndex()];
    }

    double AIMM::computeInnovation(const Eigen::VectorXd& z_meas,
                                   Eigen::VectorXd& z_pred_out,
                                   Eigen::MatrixXd& S_out) const
    {
        if (!initialized_ || filters_.empty())
        {
            z_pred_out = Eigen::VectorXd::Zero(z_meas.size());
            S_out = Eigen::MatrixXd::Identity(z_meas.size(), z_meas.size());
            return 1000.0;
        }

        const int m = z_meas.size();
        z_pred_out = Eigen::VectorXd::Zero(m);
        S_out = Eigen::MatrixXd::Zero(m, m);

        std::vector<Eigen::VectorXd> z_preds(filters_.size(), Eigen::VectorXd::Zero(m));
        std::vector<Eigen::MatrixXd> Ss(filters_.size(), Eigen::MatrixXd::Identity(m, m));

        double nis = 0.0;
        for (size_t i = 0; i < filters_.size(); ++i)
        {
            const double local_nis = filters_[i]->computeInnovation(z_meas, z_preds[i], Ss[i]);
            const double w = (i < static_cast<size_t>(mu_.size())) ? std::max(0.0, mu_(i)) : 0.0;
            z_pred_out += w * z_preds[i];
            nis += w * local_nis;
        }

        for (size_t i = 0; i < filters_.size(); ++i)
        {
            const double w = (i < static_cast<size_t>(mu_.size())) ? std::max(0.0, mu_(i)) : 0.0;
            const Eigen::VectorXd dz = z_preds[i] - z_pred_out;
            S_out += w * (Ss[i] + dz * dz.transpose());
        }

        S_out = 0.5 * (S_out + S_out.transpose());
        return std::max(nis, 0.0);
    }

    rm_radar_msgs::aimm_debugger AIMM::getDebugMsg() const
    {
        rm_radar_msgs::aimm_debugger msg;
        if (num_models_ > 0 && mu_.size() > 0) msg.mu_cv = mu_(0);
        if (num_models_ > 1 && mu_.size() > 1) msg.mu_ca = mu_(1);
        if (num_models_ > 2 && mu_.size() > 2) msg.mu_ctrv = mu_(2);
        msg.lambda = lambda_;
        msg.nis_raw = 0.0;
        msg.nis_baseline = nis_baseline_;
        msg.q_scale_alpha = std::max(1.0, std::min(q_scale_max_, lambda_ / std::max(LAMBDA_REF, 1e-6)));
        msg.active_model = static_cast<uint32_t>(getActiveModelIndex());

        return msg;
    }

}  // namespace rm_radarplugin