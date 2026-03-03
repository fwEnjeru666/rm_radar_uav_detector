#pragma once

#include <Eigen/Dense>
#include <dynamic_reconfigure/server.h>

// generated from cfg/ukf.cfg
#include <rm_track/ukfConfig.h>
#include "common.h"
#include <cmath>
#include <ros/ros.h>
#include <sstream>
#include <iomanip>

namespace rm_radarplugin
{
    class UKF
    {
    public:
        UKF() = default;
        explicit UKF(ros::NodeHandle& nh);
        ~UKF() = default;

        void initDynamicReconfigure();  // 延迟初始化 dynamic_reconfigure server
        void initialize(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0);
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

    private:
        Eigen::VectorXd state_;  // State vector
        Eigen::MatrixXd P_;      // State covariance matrix
        Eigen::MatrixXd Q_;      // Process noise covariance matrix
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
        //Q
        double q_pos_{};
        double q_vel_xy_{};
        double q_vel_z_{};
        double q_acc_xy_{};
        double q_acc_z_{};
        double q_yaw_{};
        double q_vyaw_{};
        double q_r_{};
        double q_dz_{};

        //R 
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


        void generateSigmaPoints();
        void predictSigmaPoints(const Eigen::MatrixXd& sigma_points, Eigen::MatrixXd& predicted_sigma_points);


        Eigen::VectorXd processModel(const Eigen::VectorXd& x);
        int model_type_{};

        Eigen::VectorXd computeMean(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& sigma_points, int dim);
        Eigen::MatrixXd computeCovariance(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& sigma_points, const Eigen::VectorXd& mean, int dim, int angle_idx = -1);
        void normalizeAngle(double& angle)
        {
            while (angle > M_PI) angle -= 2. * M_PI;
            while (angle < -M_PI) angle += 2. * M_PI;
        }


        //debugger
        bool debug_mode_ = false;
        double NIS_{};


        //dynamic reconfigure
        dynamic_reconfigure::Server<rm_track::ukfConfig>* ukf_cfg_srv_ = nullptr;
        dynamic_reconfigure::Server<rm_track::ukfConfig>::CallbackType ukf_cfg_cb_;
        bool ukf_initialized_ = false;
        void ukfconfigCB(rm_track::ukfConfig& config, uint32_t level);
        
        // Debug helper function
        void printQRMatrices();
        void printQRMatricesThrottled();

    public:
        // 获取当前模型类型，供外部初始化时使用
        int getModelType() const { return model_type_; }
        
        // 获取模型对应的状态维度
        static int getStateDimForModel(int model_type) {
            switch(model_type) {
                case rm_track::CV: return 6;   // [x, y, z, vx, vy, vz]
                case rm_track::CA: return 9;   // [x, y, z, vx, vy, vz, ax, ay, az]
                case rm_track::CTRV: return 6; // [x, y, z, v, yaw, yaw_rate]
                default: return 6;
            }
        }

    };
}