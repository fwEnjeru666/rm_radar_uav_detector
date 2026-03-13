#include "aimm.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace rm_radarplugin
{
    // Out-of-class definitions for static constexpr members (required pre-C++17)
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
        nh_.param("aimm/lambda_ema_alpha", lambda_ema_alpha_, 0.3);
        nh_.param("aimm/p_stay_min", p_stay_min_, 0.60);
        nh_.param("aimm/p_stay_max", p_stay_max_, 0.98);
        nh_.param("aimm/p_stay_nominal", p_stay_nominal_, 0.90);
        nh_.param("aimm/q_scale_max", q_scale_max_, 10.0);

        ROS_INFO("[AIMM] Created with %d models: CV, CA, CTRV", NUM_MODELS);
        ROS_INFO("[AIMM] Adaptive params: ema=%.2f, p_stay=[%.2f,%.2f], q_scale_max=%.1f",
                 lambda_ema_alpha_, p_stay_min_, p_stay_max_, q_scale_max_);
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
            // Compute mixed state x0j = Σ_i μ_{i|j} * x_i (in unified space)
            Eigen::VectorXd x0j = Eigen::VectorXd::Zero(OUTPUT_DIM);
            for (int i = 0; i < NUM_MODELS; ++i)
            {
                Eigen::VectorXd xi_unified = toUnifiedState(filters_[i].getState(), model_types_[i]);
                x0j += mixing_probs_(i, j) * xi_unified;
            }

            // Compute mixed covariance P0j = Σ_i μ_{i|j} * [(xi-x0j)(xi-x0j)^T + Pi]
            // Use OUTPUT_DIM x OUTPUT_DIM covariance in unified space
            Eigen::MatrixXd P0j = Eigen::MatrixXd::Zero(OUTPUT_DIM, OUTPUT_DIM);
            for (int i = 0; i < NUM_MODELS; ++i)
            {
                Eigen::VectorXd xi_unified = toUnifiedState(filters_[i].getState(), model_types_[i]);
                Eigen::VectorXd diff = xi_unified - x0j;

                // Get covariance in unified space (take top-left 6x6 block)
                Eigen::MatrixXd Pi_full = filters_[i].getCovariance();
                int dim = std::min((int)Pi_full.rows(), OUTPUT_DIM);
                Eigen::MatrixXd Pi = Eigen::MatrixXd::Identity(OUTPUT_DIM, OUTPUT_DIM) * 0.01;
                Pi.topLeftCorner(dim, dim) = Pi_full.topLeftCorner(dim, dim);

                P0j += mixing_probs_(i, j) * (Pi + diff * diff.transpose());
            }

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
        // This properly accounts for nonlinear process model effects on measurement prediction
        Eigen::VectorXd z_pred;
        Eigen::MatrixXd S;
        filters_[model_idx].computeInnovation(z_meas, z_pred, S);

        // Innovation
        Eigen::Vector3d innovation = z_meas - z_pred.head(3);

        // Gaussian likelihood: L = (2π)^{-d/2} |S|^{-1/2} exp(-0.5 * ν^T S^{-1} ν)
        double det_S = S.determinant();
        if (det_S < 1e-30) det_S = 1e-30;

        double exponent = -0.5 * innovation.transpose() * S.inverse() * innovation;
        // Clamp exponent to prevent underflow
        exponent = std::max(exponent, -500.0);

        int meas_dim = z_meas.size();
        double norm_factor = std::pow(2.0 * M_PI, -meas_dim / 2.0) * std::pow(det_S, -0.5);
        double likelihood = norm_factor * std::exp(exponent);

        return std::max(likelihood, LIKELIHOOD_FLOOR);
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
        // Combined state x = Σ_j μ_j * x_j (in unified space)
        combined_state_ = Eigen::VectorXd::Zero(OUTPUT_DIM);
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            Eigen::VectorXd xj_unified = toUnifiedState(filters_[j].getState(), model_types_[j]);
            combined_state_ += mu_(j) * xj_unified;
        }

        // Combined covariance P = Σ_j μ_j * [Pj + (xj - x)(xj - x)^T]
        combined_cov_ = Eigen::MatrixXd::Zero(OUTPUT_DIM, OUTPUT_DIM);
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            Eigen::VectorXd xj_unified = toUnifiedState(filters_[j].getState(), model_types_[j]);
            Eigen::VectorXd diff = xj_unified - combined_state_;

            Eigen::MatrixXd Pj_full = filters_[j].getCovariance();
            int dim = std::min((int)Pj_full.rows(), OUTPUT_DIM);
            Eigen::MatrixXd Pj = Eigen::MatrixXd::Identity(OUTPUT_DIM, OUTPUT_DIM) * 0.01;
            Pj.topLeftCorner(dim, dim) = Pj_full.topLeftCorner(dim, dim);

            combined_cov_ += mu_(j) * (Pj + diff * diff.transpose());
        }
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

        Eigen::Vector3d innovation = z_meas - z_pred.head(3);

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

        // 首帧：直接将 baseline 初始化为当前 NIS 值
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
    //
    // p_stay = p_stay_max - (p_stay_max - p_stay_min) * σ(λ)
    //   where σ(λ) = 1 / (1 + exp(-k*(λ/λ_ref - 2)))
    //   is a smooth sigmoid that transitions around λ ≈ 2*λ_ref
    //
    // Additionally, the off-diagonal distribution is not uniform:
    //   - From CV: more likely to switch to CA than CTRV
    //   - From CA: more likely to switch to CV (decelerate) or CTRV
    //   - From CTRV: more likely to switch to CV when λ is low
    // ============================================================
    void AIMM::adaptTPM()
    {
        // Sigmoid-based maneuver level ∈ [0, 1]
        // 0 = no maneuver, 1 = extreme maneuver
        double ratio = lambda_ / LAMBDA_REF;  // 1.0 = nominal
        double sigmoid_k = 2.0;  // steepness of transition
        double maneuver_level = 1.0 / (1.0 + std::exp(-sigmoid_k * (ratio - 2.0)));
        
        // Interpolate p_stay
        double p_stay = p_stay_max_ - (p_stay_max_ - p_stay_min_) * maneuver_level;
        double p_switch_total = 1.0 - p_stay;

        // Model indices: 0=CV, 1=CA, 2=CTRV (matching model_types_)
        // Build asymmetric off-diagonal weights based on physical reasoning:
        //
        // When maneuvering (high λ):
        //   CV should switch preferentially to CA (linear accel) or CTRV (turning)
        //   CA should switch preferentially to CTRV (if turning detected)
        //   CTRV should switch preferentially to CA (if linear accel)
        //
        // When smooth (low λ):
        //   Everything tends back toward CV
        
        // off_weight[i][j] = unnormalized weight for switching from model i to model j
        // (diagonal is excluded)
        double off_weight[NUM_MODELS][NUM_MODELS];
        
        // Weight bias: how much to favor high-maneuver models when λ is large
        double ca_bias = 1.0 + 2.0 * maneuver_level;    // CA gets more weight when maneuvering
        double ctrv_bias = 1.0 + 1.5 * maneuver_level;  // CTRV gets some weight
        double cv_bias = 1.0 + 2.0 * (1.0 - maneuver_level); // CV gets weight when smooth
        
        // From CV (index 0): switch to CA or CTRV
        off_weight[0][0] = 0;
        off_weight[0][1] = ca_bias;    // CV → CA
        off_weight[0][2] = ctrv_bias;  // CV → CTRV
        
        // From CA (index 1): switch to CV or CTRV
        off_weight[1][0] = cv_bias;    // CA → CV
        off_weight[1][1] = 0;
        off_weight[1][2] = ctrv_bias;  // CA → CTRV
        
        // From CTRV (index 2): switch to CV or CA
        off_weight[2][0] = cv_bias;    // CTRV → CV
        off_weight[2][1] = ca_bias;    // CTRV → CA
        off_weight[2][2] = 0;

        // Normalize off-diagonal per row and fill TPM
        for (int i = 0; i < NUM_MODELS; ++i)
        {
            double row_sum = 0;
            for (int j = 0; j < NUM_MODELS; ++j)
            {
                if (i != j) row_sum += off_weight[i][j];
            }
            
            for (int j = 0; j < NUM_MODELS; ++j)
            {
                if (i == j)
                    TPM_(i, j) = p_stay;
                else
                    TPM_(i, j) = p_switch_total * off_weight[i][j] / std::max(row_sum, 1e-10);
            }
        }
        
        ROS_DEBUG_THROTTLE(1, "[AIMM] adaptTPM: lambda=%.2f, level=%.2f, p_stay=%.3f", 
                           lambda_, maneuver_level, p_stay);
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
        adaptTPM();
        
        // Step 0b: Adapt process noise Q for each model
        adaptProcessNoise();

        // Step 1: Interaction (mix states across models)
        interactionStep();

        // Inject mixed states back into each filter
        for (int j = 0; j < NUM_MODELS; ++j)
        {
            Eigen::VectorXd current_state = filters_[j].getState();
            Eigen::VectorXd new_state = fromUnifiedState(mixed_states_[j], model_types_[j], current_state);
            
            // Build covariance in model-specific dimensions
            int state_dim = (int)current_state.size();
            Eigen::MatrixXd new_P = Eigen::MatrixXd::Identity(state_dim, state_dim) * 0.01;
            int dim = std::min(OUTPUT_DIM, state_dim);
            new_P.topLeftCorner(dim, dim) = mixed_covs_[j].topLeftCorner(dim, dim);
            
            // Copy remaining diagonal from original P
            Eigen::MatrixXd orig_P = filters_[j].getCovariance();
            for (int k = dim; k < state_dim; ++k)
            {
                new_P(k, k) = orig_P(k, k);
            }

            // Use lightweight setState to avoid re-initializing Q/R/weights
            filters_[j].setState(new_state, new_P);
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

        // Step 4: NOW do the actual UKF update for each model
        for (auto& f : filters_)
        {
            f.update(z_meas);
        }

        // Step 5: Combine estimates (uses post-update states)
        combineEstimates();

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

}  // namespace rm_radarplugin
