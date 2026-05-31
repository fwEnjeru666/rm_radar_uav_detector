#include "track.h"

#include <algorithm>
#include <cmath>

namespace rm_radar_lidar_detector
{

namespace
{
constexpr float kPi = 3.14159265358979323846f;
constexpr float kMinDt = 1e-3f;
constexpr float kMinYawSpeed = 0.05f;
constexpr float kMinVelocityDtFloor = 5e-3f;
constexpr float kMinPredictMaxDt = 5e-3f;
constexpr float kMinSpeedLimit = 0.5f;
constexpr double kMinGatePosBase = 0.05;
constexpr double kMinGatePosRangeK = 0.0;
constexpr double kTemporalGateMin = 0.05;
constexpr double kTemporalGateDtGain = 1.2;
constexpr double kTemporalNoPrevScore = 0.5;
constexpr double kTemporalUnassociatedPenalty = 0.2;
constexpr double kTemporalSizeDefaultScore = 0.5;
constexpr double kTemporalShapeLogSpanRatio = 0.55;
constexpr double kTemporalShapeLogSpanVolume = 0.80;
constexpr double kTemporalShapeLogSpanPca = 0.55;
constexpr double kTemporalWJump = 0.65;
constexpr double kTemporalWSize = 0.20;
constexpr double kTemporalWStreak = 0.15;
constexpr double kMarginAmbiguousScale = 0.3;
constexpr double kFinalWShape = 0.70;
constexpr double kFinalWTemporal = 0.15;
constexpr double kFinalWMargin = 0.15;
constexpr double kSecondScoreInvalid = -1.0;
}

void SingleTargetTracker::reset()
{
    assoc_state_ = AssociationState();
    state_ = State();
}

void SingleTargetTracker::setParams(const Params& params)
{
    params_ = params;
    params_.gate_distance_m = std::max(0.05f, params_.gate_distance_m);
    params_.max_lost_time_s = std::max(0.05f, params_.max_lost_time_s);
    params_.pos_alpha = static_cast<float>(clamp01(params_.pos_alpha));
    params_.vel_alpha = static_cast<float>(clamp01(params_.vel_alpha));
    params_.max_speed_mps = std::max(kMinSpeedLimit, params_.max_speed_mps);
    params_.velocity_min_dt_s = std::max(kMinVelocityDtFloor, params_.velocity_min_dt_s);
    params_.predict_max_dt_s = std::max(kMinPredictMaxDt, params_.predict_max_dt_s);
    params_.miss_velocity_decay = static_cast<float>(clamp01(params_.miss_velocity_decay));
}

double SingleTargetTracker::clamp01(double value)
{
    return std::max(0.0, std::min(1.0, value));
}

float SingleTargetTracker::wrapAngle(float angle)
{
    while (angle > kPi) angle -= 2.0f * kPi;
    while (angle < -kPi) angle += 2.0f * kPi;
    return angle;
}

void SingleTargetTracker::predictTo(const ros::Time& stamp)
{
    if (!state_.initialized || stamp <= state_.stamp)
    {
        return;
    }

    const float raw_dt = std::max(kMinDt, static_cast<float>((stamp - state_.stamp).toSec()));
    // Cap extrapolation horizon per update to avoid large jumps when timestamps jitter.
    const float dt = std::min(raw_dt, params_.predict_max_dt_s);
    state_.position += state_.velocity * dt;
    if (state_.miss_count > 0)
    {
        state_.velocity *= params_.miss_velocity_decay;
    }
    state_.stamp = stamp;

    if ((stamp - state_.last_measurement_stamp).toSec() > static_cast<double>(params_.max_lost_time_s))
    {
        state_.tracking = false;
        state_.velocity.setZero();
        state_.v_yaw = 0.0f;
        state_.accel = 0.0f;
    }
}

bool SingleTargetTracker::updateMeasurement(const Eigen::Vector3f& measurement, const ros::Time& stamp)
{
    if (!state_.initialized)
    {
        state_.initialized = true;
        state_.tracking = true;
        state_.miss_count = 0;
        state_.position = measurement;
        state_.velocity = Eigen::Vector3f::Zero();
        state_.yaw = 0.0f;
        state_.v_yaw = 0.0f;
        state_.accel = 0.0f;
        state_.stamp = stamp;
        state_.last_measurement_stamp = stamp;
        return true;
    }

    const float dt = std::max(kMinDt, static_cast<float>((stamp - state_.stamp).toSec()));
    // When sensor stamp repeats or is very close, use a safer dt for velocity/accel updates.
    const float vel_dt = std::max(params_.velocity_min_dt_s, dt);
    predictTo(stamp);
    const Eigen::Vector3f predicted = state_.position;
    const Eigen::Vector3f residual = measurement - predicted;
    const float distance = residual.norm();
    if (distance > params_.gate_distance_m)
    {
        return false;
    }

    const Eigen::Vector3f prev_velocity = state_.velocity;
    state_.position = predicted + params_.pos_alpha * residual;
    const Eigen::Vector3f measured_velocity = residual / vel_dt;
    state_.velocity = (1.0f - params_.vel_alpha) * state_.velocity + params_.vel_alpha * measured_velocity;

    const float speed = state_.velocity.norm();
    if (speed > params_.max_speed_mps)
    {
        state_.velocity *= (params_.max_speed_mps / std::max(speed, 1e-6f));
    }

    const float prev_yaw = state_.yaw;
    if (state_.velocity.head<2>().norm() > kMinYawSpeed)
    {
        state_.yaw = std::atan2(state_.velocity.y(), state_.velocity.x());
    }
    state_.v_yaw = wrapAngle(state_.yaw - prev_yaw) / vel_dt;
    state_.accel = (state_.velocity - prev_velocity).norm() / vel_dt;
    state_.miss_count = 0;
    state_.tracking = true;
    state_.stamp = stamp;
    state_.last_measurement_stamp = stamp;
    return true;
}

void SingleTargetTracker::markMiss(const ros::Time& stamp)
{
    if (!state_.initialized)
    {
        return;
    }

    predictTo(stamp);
    state_.miss_count++;
}

double SingleTargetTracker::computeTemporalScore(const Eigen::Vector3f& centroid,
                                                 double ratio,
                                                 double volume,
                                                 double pca_ratio,
                                                 double range_m,
                                                 double dt,
                                                 bool& associated,
                                                 double& normalized_jump)
{
    if (!assoc_state_.has_prev_candidate)
    {
        associated = true;
        normalized_jump = 0.0;
        return kTemporalNoPrevScore;
    }

    const double jump = static_cast<double>((centroid - assoc_state_.prev_centroid).norm());
    const double gate = std::max(
        kTemporalGateMin,
        assoc_params_.gate_pos_base + assoc_params_.gate_pos_range_k * range_m + kTemporalGateDtGain * dt);
    associated = jump <= gate;
    normalized_jump = jump / gate;

    double jump_score = clamp01(1.0 - normalized_jump);
    if (!associated)
    {
        jump_score *= kTemporalUnassociatedPenalty;
    }

    double size_score = kTemporalSizeDefaultScore;
    if (assoc_state_.prev_ratio > 1e-6 && assoc_state_.prev_volume > 1e-6 && assoc_state_.prev_pca_ratio > 1e-6 &&
        ratio > 1e-6 && volume > 1e-6 && pca_ratio > 1e-6)
    {
        const double ratio_delta = std::abs(std::log(ratio) - std::log(assoc_state_.prev_ratio));
        const double volume_delta = std::abs(std::log(volume) - std::log(assoc_state_.prev_volume));
        const double pca_delta = std::abs(std::log(pca_ratio) - std::log(assoc_state_.prev_pca_ratio));
        const double ratio_score = clamp01(1.0 - ratio_delta / kTemporalShapeLogSpanRatio);
        const double volume_score = clamp01(1.0 - volume_delta / kTemporalShapeLogSpanVolume);
        const double pca_score = clamp01(1.0 - pca_delta / kTemporalShapeLogSpanPca);
        size_score = 0.45 * ratio_score + 0.35 * volume_score + 0.20 * pca_score;
    }

    const double streak_norm_den = static_cast<double>(std::max(1, assoc_params_.lock_min_streak + 2));
    const double streak_score = clamp01(static_cast<double>(assoc_state_.streak) / streak_norm_den);
    return clamp01(kTemporalWJump * jump_score + kTemporalWSize * size_score + kTemporalWStreak * streak_score);
}

double SingleTargetTracker::computeMarginScore(double best_score, double second_score, bool& ambiguous) const
{
    if (second_score < 0.0)
    {
        ambiguous = false;
        return 1.0;
    }

    const double margin = std::max(0.0, best_score - second_score);
    ambiguous = margin < (kMarginAmbiguousScale * assoc_params_.margin_norm);
    return clamp01(margin / std::max(1e-3, assoc_params_.margin_norm));
}

double SingleTargetTracker::computeFinalConfidence(double shape_score,
                                                   double temporal_score,
                                                   double margin_score) const
{
    return clamp01(kFinalWShape * shape_score + kFinalWTemporal * temporal_score + kFinalWMargin * margin_score);
}

SingleTargetTracker::AssociationResult SingleTargetTracker::updateAssociation(
    const Eigen::Vector3f* centroid,
    double ratio,   
    double volume,
    double pca_ratio,
    double shape_score,
    double second_score,
    double range_m,
    double dt)
{
    AssociationResult result;
    const bool locked_before = assoc_state_.locked;

    assoc_params_.conf_on = clamp01(assoc_params_.conf_on);
    assoc_params_.conf_keep = clamp01(assoc_params_.conf_keep);
    assoc_params_.lock_min_streak = std::max(1, assoc_params_.lock_min_streak);
    assoc_params_.gate_pos_base = std::max(kMinGatePosBase, assoc_params_.gate_pos_base);
    assoc_params_.gate_pos_range_k = std::max(kMinGatePosRangeK, assoc_params_.gate_pos_range_k);
    if (assoc_params_.conf_keep > assoc_params_.conf_on)
    {
        assoc_params_.conf_keep = assoc_params_.conf_on;
    }
    if (assoc_params_.margin_norm < 1e-3)
    {
        assoc_params_.margin_norm = 1e-3;
    }
    if (assoc_params_.max_miss_before_unlock < 0)
    {
        assoc_params_.max_miss_before_unlock = 0;
    }

    const double safe_dt = std::max(1e-3, dt);
    if (centroid)
    {
        bool associated = false;
        double normalized_jump = std::numeric_limits<double>::infinity();
        const double shape = clamp01(shape_score);
        const double temporal = computeTemporalScore(*centroid, ratio, volume, pca_ratio, std::max(0.0, range_m), safe_dt,
                                                     associated, normalized_jump);
        bool ambiguous = false;
        const double margin = computeMarginScore(shape, second_score, ambiguous);
        const double confidence = computeFinalConfidence(shape, temporal, margin);

        if (!assoc_state_.has_prev_candidate)
        {
            assoc_state_.streak = 1;
        }
        else
        {
            assoc_state_.streak = associated ? (assoc_state_.streak + 1) : 1;
        }

        assoc_state_.prev_centroid = *centroid;
        assoc_state_.prev_ratio = ratio;
        assoc_state_.prev_volume = volume;
        assoc_state_.prev_pca_ratio = pca_ratio;
        assoc_state_.has_prev_candidate = true;
        assoc_state_.last_confidence = confidence;

        if (assoc_state_.locked)
        {
            if (confidence >= assoc_params_.conf_keep)
            {
                result.accepted = true;
                assoc_state_.miss_count = 0;
            }
            else
            {
                assoc_state_.miss_count++;
                if (assoc_state_.miss_count > assoc_params_.max_miss_before_unlock)
                {
                    assoc_state_.locked = false;
                    assoc_state_.miss_count = 0;
                }
            }
        }
        else if (confidence >= assoc_params_.conf_on &&
                 assoc_state_.streak >= assoc_params_.lock_min_streak)
        {
            assoc_state_.locked = true;
            assoc_state_.miss_count = 0;
            result.accepted = true;
        }
        else if (!assoc_params_.publish_only_when_locked &&
                 confidence >= assoc_params_.conf_keep)
        {
            result.accepted = true;
        }

        result.associated = associated;
        result.ambiguous = ambiguous;
        result.normalized_jump = normalized_jump;
        result.confidence = confidence;
    }
    else
    {
        assoc_state_.streak = 0;
        if (assoc_state_.locked)
        {
            assoc_state_.miss_count++;
            if (assoc_state_.miss_count > assoc_params_.max_miss_before_unlock)
            {
                assoc_state_.locked = false;
                assoc_state_.miss_count = 0;
            }
        }
        result.confidence = assoc_state_.last_confidence;
    }

    result.lock_lost = locked_before && !assoc_state_.locked;
    result.lock_gained = !locked_before && assoc_state_.locked;
    result.locked = assoc_state_.locked;
    result.streak = assoc_state_.streak;
    return result;
}

}  // namespace rm_radar_lidar_detector
