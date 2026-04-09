#pragma once

#include <Eigen/Dense>
#include <ros/ros.h>
#include <vector>
#include <string>
#include "ukf.h"
#include <rm_track/aimmConfig.h>
#include <dynamic_reconfigure/server.h>
#include <rm_radar_msgs/aimm_debugger.h>

namespace rm_radarplugin
{

class AIMM
{
public:
    AIMM() = default;
    explicit AIMM(ros::NodeHandle& nh, const std::vector<std::shared_ptr<BaseModel>>& models);;
    ~AIMM() = default;

    void initialize(const Eigen::Vector3d& z_meas);

    void setDt(double dt);
    void predict();
    void update(const Eigen::VectorXd& z_meas);

    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;
    bool isInitialized() const { return initialized_; }

    // R matrix interface (same as UKF, applies to all models)
    void setMeasurementNoise(const Eigen::MatrixXd& R);
    Eigen::MatrixXd getMeasurementNoise() const;
    Eigen::MatrixXd getBaseMeasurementNoise() const;

    // Model info
    int getActiveModelIndex() const;
    Eigen::VectorXd getModelProbabilities() const { return mu_; }
    double getManeuverIndicator() const { return lambda_; }
    rm_radar_msgs::aimm_debugger getDebugMsg() const;
    double computeInnovation(const Eigen::VectorXd& z_meas,
                             Eigen::VectorXd& z_pred_out,
                             Eigen::MatrixXd& S_out) const;
private:
    int num_models_ {0};
    
    ros::NodeHandle nh_;
    bool initialized_ = false;

    // Individual UKF filters
    std::vector<std::shared_ptr<UKF>> filters_;
    std::vector<std::string> model_names_;

    // Dynamic Reconfigure Server for AIMM parameters
    std::shared_ptr<dynamic_reconfigure::Server<rm_track::aimmConfig>> aimm_cfg_server_;
    void aimmconfigCB(rm_track::aimmConfig& config, uint32_t level);

    std::string getActiveModelName() const;

    // Adaptive params matching dynamic_reconfigure
    // IMM Probabilities
    Eigen::VectorXd mu_;           // model probabilities: [NUM_MODELS x 1]

    // Markov transition probability matrix π_ij  (ADAPTIVE — changes every step)
    Eigen::MatrixXd TPM_;

    // Mixed states/covariances after interaction step
    std::vector<Eigen::VectorXd> mixed_states_;
    std::vector<Eigen::MatrixXd> mixed_covs_;

    // Mixing probabilities μ_{i|j}
    Eigen::MatrixXd mixing_probs_;

    // Current smoothed maneuver indicator
    double lambda_ = 1.0;
    // Parameter for alpha limit
    double LAMBDA_REF = 5.0;

    int OUTPUT_DIM = 9;

    // Adaptive maneuver constants base
    double prob_maneuver_detection_ = 0.5;
    double prob_maneuver_confirmation_ = 0.5;
    double prob_maneuver_switch_ = 0.5;
    double prob_mix_switch_ca_ = 0.5;
    double prob_mix_switch_ctrv_ = 0.5;

    // =================== Adaptive machinery ===================
    
    // Maneuver indicator λ(k): ratio of current NIS to baseline NIS
    // λ ≈ 1.0 → smooth motion (NIS matches expectation)
    // λ >> 1.0 → aggressive maneuver (NIS spikes above baseline)
    
    // Exponential moving average smoothing factor for λ
    // λ(k) = α_ema * NIS_raw + (1 - α_ema) * λ(k-1)
    double lambda_ema_alpha_{0.3};
    
    // Reference λ: the expected λ under H0 (no maneuver)
    // Since λ is now a NIS ratio (NIS_current / NIS_baseline),
    // the expected value under no maneuver is 1.0
    double lambda_ref_{1.0};  // Expected λ when model matches well (NIS ≈ baseline)
    
    // TPM adaptation parameters
    double p_stay_min_{0.60};    // minimum stay probability (extreme maneuver)
    double p_stay_max_{0.98};    // maximum stay probability (smooth motion)
    double p_stay_nominal_{0.90};// nominal (starting) stay probability
    
    // Q scaling parameters
    double q_scale_max_{10.0};   // max Q multiplier during heavy maneuver
    
    // Maneuver detection: per-model NIS from last update
    std::vector<double> model_NIS_;

    // Adaptive NIS baseline: tracks the expected NIS when filter is well-matched
    // This makes λ self-calibrating — if Q is large and NIS is always small,
    // the baseline drops accordingly, and only deviations FROM the baseline
    // trigger maneuver detection.
    double nis_baseline_{};          // 0 表示未初始化，首帧会一步校准
    double nis_baseline_alpha_{};   // baseline EMA 速度（越大越快追踪）
    bool nis_baseline_initialized_{false};  // 首帧标志：第一次收到 NIS 时直接赋值
    
    // Adaptive
    
    Eigen::Vector3d last_innovation_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d last_velocity_ = Eigen::Vector3d::Zero();
    double sigmoid_k_{};  // Sigmoid steepness for TPM adaptation
    double maneuver_gain_{}; // Additional gain factor for TPM adaptation (optional, can be tuned to make TPM more or less sensitive to λ)
    void adaptTPM(const Eigen::Vector3d& innovation, const Eigen::Vector3d& vel); // Adjust TPM_ based on λ


    void adaptProcessNoise();    // Scale each filter's Q based on λ
    void updateManeuverIndicator(const Eigen::VectorXd& z_meas);
    double computeNIS(int model_idx, const Eigen::VectorXd& z_meas);

    void interactionStep();
    void computeMixingProbabilities();
    double computeLikelihood(int model_idx, const Eigen::VectorXd& z_meas);
    void updateModelProbabilities(const Eigen::VectorXd& z_meas);
    void combineEstimates();

    // Combined output
    Eigen::VectorXd combined_state_;
    Eigen::MatrixXd combined_cov_;

    // Likelihood lower bound to prevent numerical issues
    static constexpr double LIKELIHOOD_FLOOR = 1e-300;
};

}  // namespace rm_radarplugin
