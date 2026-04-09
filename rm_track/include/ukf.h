#pragma once

#include <Eigen/Dense>
#include <dynamic_reconfigure/server.h>

#include <rm_track/ukfConfig.h>
#include "common.h"
#include <cmath>
#include <ros/ros.h>
#include <sstream>
#include <iomanip>

#include "motion_models_define.h"
namespace rm_radarplugin
{
    class UKF
    {
    public:
        UKF() = default;
        explicit UKF(ros::NodeHandle& nh, std::shared_ptr<BaseModel> model);
        void initialize(const Eigen::Vector3d& z_meas);
        std::shared_ptr<BaseModel> getModel() const { return model_;}
        void ukf_params_init(rm_track::ukfConfig& config, uint32_t level);
        void predict();
        void update(const Eigen::VectorXd& z_meas);
        Eigen::VectorXd getState() const;
        Eigen::MatrixXd getCovariance() const;
        void setDt(double dt) { dt_ = dt; }
        bool isInitialized() const { return ukf_initialized_; }
        void setMeasurementNoise(const Eigen::MatrixXd& R) { R_ = R; }
        Eigen::MatrixXd getMeasurementNoise() const { return R_; }
        Eigen::MatrixXd getBaseMeasurementNoise() const { return R_base_; }

        // Set a new motion model
        void setModel(std::shared_ptr<BaseModel> model);


        // Q scale interface for AIMM adaptive scaling
        void setQScale(double scale) { q_scale_ = scale; }
        double getQScale() const { return q_scale_; }

        // Lightweight state injection for AIMM interaction step (no re-init of Q/R/weights)
        void setState(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
            state_ = x;
            P_ = P;
        }

        // Access the innovation covariance S_ from last update (for AIMM likelihood)
        Eigen::MatrixXd getInnovationCovariance() const { return S_; }
        Eigen::VectorXd getPredictedMeasurement() const { return z_pred_; }

        // Compute innovation statistics (z_pred, S, NIS) from current predicted sigma points
        // WITHOUT modifying state. Call AFTER predict(), BEFORE update().
        // Returns NIS = ν^T S^{-1} ν where ν = z - z_pred
        double computeInnovation(const Eigen::VectorXd& z_meas,
                                 Eigen::VectorXd& z_pred_out,
                                 Eigen::MatrixXd& S_out) const;

        void ukfconfigCB(rm_track::ukfConfig& config, uint32_t level);
        
        // Debug helper function
        void printQRMatrices();
        void printQRMatricesThrottled();

    private:
        double q_scale_{1.0}; // Adaptive process noise multiplier
        std::shared_ptr<BaseModel> model_;

        Eigen::VectorXd state_;  // State vector
        Eigen::MatrixXd P_;      // State covariance matrix
        Eigen::MatrixXd R_;      // Measurement noise covariance matrix (per-frame, may be scaled)
        Eigen::MatrixXd R_base_; // Base measurement noise from dynamic_reconfigure

        Eigen::MatrixXd Xsig_; // Sigma points matrix
        Eigen::MatrixXd Xsig_pred_; // Predicted sigma points matrix
        Eigen::MatrixXd weights_;     // Weights for sigma points
        Eigen::MatrixXd Zsig_;       // Sigma points in measurement space
        Eigen::VectorXd z_pred_;    // Predicted measurement mean
        Eigen::VectorXd z_meas_;     // Measurement vector
        Eigen::MatrixXd S_;         // Measurement covariance matrix
        Eigen::MatrixXd Tc_;        // Cross correlation matrix
        Eigen::MatrixXd K_;        // Kalman gain matrix

        ros::NodeHandle nh_;

        // R 
        double r_pos_xy_{};
        double r_pos_z_{};
        double r_yaw_{};

        double dt_{};

        int state_dim_{};
        int meas_dim_{};
        int sigma_point_count_{};
        double lambda_{};

        //means and covariances
        Eigen::MatrixXd weights_m_{};
        Eigen::MatrixXd weights_c_{};

        void weightsInit(std::shared_ptr<BaseModel> model);
        void generateSigmaPoints();
        
        Eigen::VectorXd computeMean(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& sigma_points, int dim) const;
        Eigen::MatrixXd computeCovariance(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& sigma_points, const Eigen::VectorXd& mean, int dim, int angle_idx = -1) const;
        void normalizeAngle(double& angle) const
        {
            while (angle > M_PI) angle -= 2. * M_PI;
            while (angle < -M_PI) angle += 2. * M_PI;
        }

        //debugger
        bool debug_mode_ = false;
        double NIS_{};

        //dynamic reconfigure
        std::unique_ptr<dynamic_reconfigure::Server<rm_track::ukfConfig>> ukf_cfg_srv_;
        dynamic_reconfigure::Server<rm_track::ukfConfig>::CallbackType ukf_cfg_cb_;
        bool ukf_initialized_ = false;
    };
}