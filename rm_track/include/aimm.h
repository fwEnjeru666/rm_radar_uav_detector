#pragma once

#include <Eigen/Dense>
#include <ros/ros.h>
#include <vector>
#include <string>
#include <dynamic_reconfigure/server.h>
#include <rm_track/ukfConfig.h>
#include "ukf.h"
#include "common.h"

namespace rm_radarplugin
{

/**
 * @brief AIMM - Adaptive Interacting Multiple Model estimator
 * 
 * Key difference from standard IMM: the transition probability matrix (TPM)
 * and process noise (Q) are **not fixed constants**. They are adapted in
 * real-time based on a maneuver indicator λ(k) derived from the combined
 * normalised innovation squared (NIS).
 * 
 * Adaptive mechanisms:
 *   A) TPM adaptation — when λ is large (aggressive maneuver), p_stay drops
 *      and switching probabilities toward CA/CTRV increase. When λ is small
 *      (smooth motion), p_stay rises and the system prefers CV.
 *   B) Q scaling — each model's process noise Q is scaled by a factor
 *      α(k) = clamp(λ/λ_ref, 1, α_max) so that filters open up during
 *      maneuvers and tighten during steady motion.
 * 
 * Pipeline per timestep:
 *   0. Adapt TPM and Q based on previous-step maneuver indicator
 *   1. Interaction (mix states across models)
 *   2. Predict (each model independently, with adapted Q)
 *   3. Update (each model independently)
 *   4. Compute model likelihoods
 *   5. Update model probabilities
 *   6. Combine output state as probability-weighted sum
 *   7. Update maneuver indicator for next step
 */
class AIMM
{
public:
    AIMM() = default;
    explicit AIMM(ros::NodeHandle& nh);
    ~AIMM() = default;

    void initDynamicReconfigure();
    void initialize(const Eigen::Vector3d& z_meas);

    void setDt(double dt);
    void predict();
    void update(const Eigen::Vector3d& z_meas);

    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;
    bool isInitialized() const { return initialized_; }

    // R matrix interface (same as UKF, applies to all models)
    void setMeasurementNoise(const Eigen::MatrixXd& R);
    Eigen::MatrixXd getMeasurementNoise() const;
    Eigen::MatrixXd getBaseMeasurementNoise() const;

    // Model info
    int getActiveModelIndex() const;
    std::string getActiveModelName() const;
    Eigen::VectorXd getModelProbabilities() const { return mu_; }
    int getModelType() const;  // returns type of most probable model
    double getManeuverIndicator() const { return lambda_; }

    // For compatibility with existing code
    static int getStateDimForModel(int model_type) {
        return UKF::getStateDimForModel(model_type);
    }

private:
    static constexpr int NUM_MODELS = 3;  // CV, CA, CTRV
    
    ros::NodeHandle nh_;
    bool initialized_ = false;

    // Individual UKF filters
    std::vector<UKF> filters_;
    std::vector<int> model_types_;       // rm_track::CV, CA, CTRV
    std::vector<std::string> model_names_;

    // Model probabilities μ_j (sum = 1)
    Eigen::VectorXd mu_;

    // Markov transition probability matrix π_ij  (ADAPTIVE — changes every step)
    Eigen::MatrixXd TPM_;

    // Mixed states/covariances after interaction step
    std::vector<Eigen::VectorXd> mixed_states_;
    std::vector<Eigen::MatrixXd> mixed_covs_;

    // Mixing probabilities μ_{i|j}
    Eigen::MatrixXd mixing_probs_;

    // Output state dimension (unified to 6: [x, y, z, vx, vy, vz])
    static constexpr int OUTPUT_DIM = 6;

    // =================== Adaptive machinery ===================
    
    // Maneuver indicator λ(k): weighted NIS from combined innovation
    // Small λ → smooth motion, large λ → aggressive maneuver
    double lambda_{0.0};
    
    // Exponential moving average smoothing factor for λ
    // λ(k) = α_ema * NIS_raw + (1 - α_ema) * λ(k-1)
    double lambda_ema_alpha_{0.3};
    
    // Reference λ: the expected NIS under H0 (no maneuver)
    // For 3D measurements, NIS ~ chi²(3), so E[NIS] = 3.0
    static constexpr double LAMBDA_REF = 3.0;
    
    // TPM adaptation parameters
    double p_stay_min_{0.60};    // minimum stay probability (extreme maneuver)
    double p_stay_max_{0.98};    // maximum stay probability (smooth motion)
    double p_stay_nominal_{0.90};// nominal (starting) stay probability
    
    // Q scaling parameters
    double q_scale_max_{10.0};   // max Q multiplier during heavy maneuver
    // Q_base_[i] stores the original Q from dynamic_reconfigure for each filter
    std::vector<Eigen::MatrixXd> Q_base_;
    
    // Maneuver detection: per-model NIS from last update
    std::vector<double> model_NIS_;
    
    // Adaptive methods
    void adaptTPM();             // Adjust TPM_ based on λ
    void adaptProcessNoise();    // Scale each filter's Q based on λ
    void updateManeuverIndicator(const Eigen::Vector3d& z_meas);
    double computeNIS(int model_idx, const Eigen::Vector3d& z_meas);
    
    // =================== Standard IMM steps ===================
    void interactionStep();
    void computeMixingProbabilities();
    double computeLikelihood(int model_idx, const Eigen::Vector3d& z_meas);
    void updateModelProbabilities(const Eigen::Vector3d& z_meas);
    void combineEstimates();

    // Map model state to unified 6D state [x,y,z,vx,vy,vz]
    Eigen::VectorXd toUnifiedState(const Eigen::VectorXd& state, int model_type) const;
    // Map unified 6D state back to model-specific state
    Eigen::VectorXd fromUnifiedState(const Eigen::VectorXd& unified, int model_type, 
                                      const Eigen::VectorXd& original) const;

    // Combined output
    Eigen::VectorXd combined_state_;
    Eigen::MatrixXd combined_cov_;

    // Likelihood lower bound to prevent numerical issues
    static constexpr double LIKELIHOOD_FLOOR = 1e-300;
};

}  // namespace rm_radarplugin
