#pragma once

#include <Eigen/Dense>
#include <ros/ros.h>
#include <limits>

namespace rm_radar_lidar_detector
{

class SingleTargetTracker
{
public:
    struct Params
    {
        float gate_distance_m = 1.5f;
        float max_lost_time_s = 0.8f;
        float pos_alpha = 0.65f;
        float vel_alpha = 0.45f;
        float max_speed_mps = 15.0f;
        // Lower bound for dt used in velocity/accel update to avoid tiny-dt spikes.
        float velocity_min_dt_s = 0.03f;
        // Upper bound for single-step prediction horizon to avoid runaway extrapolation.
        float predict_max_dt_s = 0.08f;
        // Velocity decay applied on each miss frame, in [0, 1].
        float miss_velocity_decay = 0.85f;
    };

    struct AssociationParams
    {
        double conf_on = 0.68;
        double conf_keep = 0.52;
        int lock_min_streak = 3;
        double gate_pos_base = 0.60;
        double gate_pos_range_k = 0.03;
        double margin_norm = 0.35;
        int max_miss_before_unlock = 2;
        bool publish_only_when_locked = true;
    };

    struct AssociationState
    {
        bool has_prev_candidate = false;
        bool locked = false;
        int streak = 0;
        int miss_count = 0;
        Eigen::Vector3f prev_centroid = Eigen::Vector3f::Zero();
        double prev_ratio = 0.0;
        double prev_volume = 0.0;
        double prev_pca_ratio = 0.0;
        double last_confidence = 0.0;
    };

    struct AssociationResult
    {
        bool accepted = false;
        bool associated = false;
        bool ambiguous = false;
        bool lock_lost = false;
        bool lock_gained = false;
        bool locked = false;
        int streak = 0;
        double confidence = 0.0;
        double normalized_jump = std::numeric_limits<double>::infinity();
    };

    struct State
    {
        bool initialized = false;
        bool tracking = false;
        int id = 1;
        int miss_count = 0;
        ros::Time stamp;
        ros::Time last_measurement_stamp;
        Eigen::Vector3f position = Eigen::Vector3f::Zero();
        Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
        float yaw = 0.0f;
        float v_yaw = 0.0f;
        float accel = 0.0f;
    };

    void setParams(const Params& params);
    const Params& getParams() const { return params_; }
    const State& getState() const { return state_; }
    void setAssociationParams(const AssociationParams& params) { assoc_params_ = params; }
    const AssociationParams& getAssociationParams() const { return assoc_params_; }
    const AssociationState& getAssociationState() const { return assoc_state_; }

    void reset();
    bool updateMeasurement(const Eigen::Vector3f& measurement, const ros::Time& stamp);
    void markMiss(const ros::Time& stamp);
    AssociationResult updateAssociation(const Eigen::Vector3f* centroid,
                                        double ratio,
                                        double volume,
                                        double pca_ratio,
                                        double shape_score,
                                        double second_score,
                                        double range_m,
                                        double dt);
    bool hasTrack() const { return state_.initialized && state_.tracking; }

private:
    static double clamp01(double value);
    static float wrapAngle(float angle);
    void predictTo(const ros::Time& stamp);
    double computeTemporalScore(const Eigen::Vector3f& centroid,
                                double ratio,
                                double volume,
                                double pca_ratio,
                                double range_m,
                                double dt,
                                bool& associated,
                                double& normalized_jump);
    double computeMarginScore(double best_score, double second_score, bool& ambiguous) const;
    double computeFinalConfidence(double shape_score, double temporal_score, double margin_score) const;

    Params params_;
    AssociationParams assoc_params_;
    AssociationState assoc_state_;
    State state_;
};

}  // namespace rm_radar_lidar_detector
