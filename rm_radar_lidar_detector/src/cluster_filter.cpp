#include "cluster_filter.h"

#include <algorithm>
#include <cmath>
#include <Eigen/Eigenvalues>

namespace rm_radar_lidar_detector
{
    namespace
    {
        constexpr double kInvalidBestScore = -1.0;
        constexpr double kClampMin = 0.0;
        constexpr double kClampMax = 1.0;
        constexpr double kMinSpan = 1e-6;
        constexpr double kMinDimEpsilon = 1e-3;
        constexpr double kInvalidFlatRatio = 1000.0;
        constexpr double kInvalidPcaRatio = 1000.0;
        constexpr double kShapeWeightRatio = 0.45;
        constexpr double kShapeWeightVolume = 0.35;
        constexpr double kShapeWeightPca = 0.20;
    }

    bool ClusterFilter::validateMetrics(int point_count, double volume, double ratio, double pca_ratio, float center_z) const
    {
        if (point_count < params_.min_n || point_count > params_.max_n) {
            return false;
        }
        if (volume < params_.min_v || volume > params_.max_v) {
            return false;
        }
        if (ratio < params_.min_r || ratio > params_.max_r) {
            return false;
        }
        if (pca_ratio < params_.min_pca_r || pca_ratio > params_.max_pca_r) {
            return false;
        }

        const float min_h = std::min(params_.min_h, params_.max_h);
        const float max_h = std::max(params_.min_h, params_.max_h);
        return center_z >= min_h && center_z <= max_h;
    }

    bool ClusterFilter::validate(
        const PointCloudPtr& cluster,
        const pcl::PointXYZ& min_pt,
        const pcl::PointXYZ& max_pt)
    {
        if (!cluster || cluster->empty()) {
            return false;
        }
        
        const int point_count = static_cast<int>(cluster->size());
        const double volume = getVolume(min_pt, max_pt);
        const double ratio = getRatio(min_pt, max_pt);
        const double pca_ratio = getPcaRatio(cluster);
        const float center_z = 0.5f * (min_pt.z + max_pt.z);
        return validateMetrics(point_count, volume, ratio, pca_ratio, center_z);
    }

    ClusterFilter::PointCloudPtr ClusterFilter::findBest(
        const ClusterMap& clusters,
        BBox3D& best_bbox,
        Eigen::Vector4f& best_centroid,
        std::vector<ClusterDebugInfo>* debug_infos)
    {
        PointCloudPtr best_cluster = nullptr;
        double best_score = kInvalidBestScore;
        int best_cluster_index = -1;

        if (debug_infos) {
            debug_infos->clear();
            debug_infos->reserve(clusters.size());
        }

        auto clamp01 = [](double v) {
            return std::max(kClampMin, std::min(kClampMax, v));
        };

        // Scoring only uses relatively stable object-shape terms:
        // ratio (dimensionless) + volume.
        // point_count/height are hard gates in validateMetrics().
        const double min_r_safe = std::max(params_.min_r, kMinSpan);
        const double max_r_safe = std::max(params_.max_r, min_r_safe + kMinSpan);
        const double min_v_safe = std::max(params_.min_v, kMinSpan);
        const double max_v_safe = std::max(params_.max_v, min_v_safe + kMinSpan);
        const double min_pca_safe = std::max(params_.min_pca_r, kMinSpan);
        const double max_pca_safe = std::max(params_.max_pca_r, min_pca_safe + kMinSpan);

        // Use geometric center + log span, more robust than linear center.
        const double log_r_mid = 0.5 * (std::log(min_r_safe) + std::log(max_r_safe));
        const double log_r_half = std::max(kMinSpan, 0.5 * (std::log(max_r_safe) - std::log(min_r_safe)));
        const double log_v_mid = 0.5 * (std::log(min_v_safe) + std::log(max_v_safe));
        const double log_v_half = std::max(kMinSpan, 0.5 * (std::log(max_v_safe) - std::log(min_v_safe)));
        const double log_pca_mid = 0.5 * (std::log(min_pca_safe) + std::log(max_pca_safe));
        const double log_pca_half = std::max(kMinSpan, 0.5 * (std::log(max_pca_safe) - std::log(min_pca_safe)));
        
        for (size_t cluster_idx = 0; cluster_idx < clusters.size(); ++cluster_idx) {
            const auto& cluster = clusters[cluster_idx];
            
            if (!cluster || cluster->empty()) {
                continue;
            }

            ClusterDebugInfo info;
            info.cluster_index = static_cast<int>(cluster_idx);
            info.point_count = static_cast<int>(cluster->size());
            info.centroid = getCentroid(cluster);
            
            // Compute bounding box
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);
            info.bbox.update(min_pt, max_pt);
            info.volume = getVolume(min_pt, max_pt);
            info.ratio = getRatio(min_pt, max_pt);
            info.pca_ratio = getPcaRatio(cluster);

            const double point_count = static_cast<double>(info.point_count);
            const double ratio = info.ratio;
            const double volume = info.volume;
            const double pca_ratio = info.pca_ratio;
            const float center_z_f = 0.5f * (min_pt.z + max_pt.z);
            (void)point_count;

            const double ratio_score = clamp01(1.0 - std::abs(std::log(std::max(ratio, kMinSpan)) - log_r_mid) / log_r_half);
            const double volume_score = clamp01(1.0 - std::abs(std::log(std::max(volume, kMinSpan)) - log_v_mid) / log_v_half);
            const double pca_score = clamp01(1.0 - std::abs(std::log(std::max(pca_ratio, kMinSpan)) - log_pca_mid) / log_pca_half);
            const double score =
                kShapeWeightRatio * ratio_score +
                kShapeWeightVolume * volume_score +
                kShapeWeightPca * pca_score;
            info.score = score;
            
            // Validate cluster
            if (!validateMetrics(info.point_count, volume, ratio, pca_ratio, center_z_f)) {
                info.valid = false;
                if (debug_infos) {
                    debug_infos->push_back(info);
                }
                continue;
            }
            info.valid = true;
            if (debug_infos) {
                debug_infos->push_back(info);
            }

            if (score > best_score) {
                best_score = score;
                best_cluster = cluster;
                best_cluster_index = static_cast<int>(cluster_idx);
                best_bbox.update(min_pt, max_pt);
                best_centroid = info.centroid;
            }
        }

        if (debug_infos && best_cluster_index >= 0) {
            for (auto& info : *debug_infos) {
                if (info.valid && info.cluster_index == best_cluster_index) {
                    info.is_best = true;
                    break;
                }
            }
        }
        
        return best_cluster;
    }

    double ClusterFilter::getVolume(const pcl::PointXYZ& min_pt, const pcl::PointXYZ& max_pt)
    {
        double dx = max_pt.x - min_pt.x;
        double dy = max_pt.y - min_pt.y;
        double dz = max_pt.z - min_pt.z;
        return dx * dy * dz;
    }

    double ClusterFilter::getRatio(const pcl::PointXYZ& min_pt, const pcl::PointXYZ& max_pt)
    {
        double dx = max_pt.x - min_pt.x;
        double dy = max_pt.y - min_pt.y;
        double dz = max_pt.z - min_pt.z;

        const double max_dim = std::max({dx, dy, dz});
        const double min_dim = std::min({dx, dy, dz});
        if (min_dim < kMinDimEpsilon)
        {
            return kInvalidFlatRatio;
        }
        return max_dim / min_dim;
    }

    Eigen::Vector4f ClusterFilter::getCentroid(const PointCloudPtr& cluster)
    {
        Eigen::Vector4f centroid;
        if (cluster && !cluster->empty()) {
            pcl::compute3DCentroid(*cluster, centroid);
        } else {
            centroid = Eigen::Vector4f::Zero();
        }
        return centroid;
    }

    double ClusterFilter::getPcaRatio(const PointCloudPtr& cluster)
    {
        if (!cluster || cluster->size() < 3)
        {
            return kInvalidPcaRatio;
        }

        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
        if (pcl::computeMeanAndCovarianceMatrix(*cluster, covariance, centroid) <= 0)
        {
            return kInvalidPcaRatio;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        if (solver.info() != Eigen::Success)
        {
            return kInvalidPcaRatio;
        }

        // SelfAdjointEigenSolver returns ascending eigenvalues.
        const auto eigenvalues = solver.eigenvalues();
        const double lambda_min = std::max(kMinSpan, static_cast<double>(eigenvalues[0]));
        const double lambda_max = std::max(lambda_min, static_cast<double>(eigenvalues[2]));
        return std::sqrt(lambda_max / lambda_min);
    }


CandidateSelectionResult CandidateSelector::select(const ClusterMap& clusters,
                                                   std::vector<ClusterDebugInfo>& debug_infos,
                                                   const CandidateSelectionContext& context) const
{
    if (context.use_track_position)
    {
        return selectByTrackPosition(clusters, debug_infos, context);
    }
    return selectByShape(debug_infos);
}

CandidateSelectionResult CandidateSelector::selectByShape(std::vector<ClusterDebugInfo>& debug_infos) const
{
    CandidateSelectionResult result;
    double best_score = -1.0;

    for (const auto& info : debug_infos)
    {
        if (!info.valid)
        {
            continue;
        }

        ++result.valid_count;
        if (info.is_best || info.score > best_score)
        {
            if (result.found)
            {
                result.second_score = std::max(result.second_score, best_score);
            }
            result.info = info;
            best_score = info.score;
            result.found = true;
        }
        else
        {
            result.second_score = std::max(result.second_score, info.score);
        }
    }

    return result;
}

CandidateSelectionResult CandidateSelector::selectByTrackPosition(const ClusterMap& clusters,
                                                                  std::vector<ClusterDebugInfo>& debug_infos,
                                                                  const CandidateSelectionContext& context) const
{
    CandidateSelectionResult result;
    const float gate = std::max(0.05f, context.track_gate_distance);
    const double shape_weight = std::max(0.0, context.shape_weight);
    const double temporal_weight = std::max(0.0, context.temporal_weight);
    const double weight_sum = std::max(1e-6, shape_weight + temporal_weight);
    const double normalized_shape_weight = shape_weight / weight_sum;
    const double normalized_temporal_weight = temporal_weight / weight_sum;
    double selected_combined_score = -1.0;

    for (auto& info : debug_infos)
    {
        info.is_best = false;
        if (!info.valid)
        {
            continue;
        }

        ++result.valid_count;
        if (info.cluster_index < 0 || info.cluster_index >= static_cast<int>(clusters.size()) ||
            !clusters[info.cluster_index] || clusters[info.cluster_index]->empty())
        {
            continue;
        }

        const Eigen::Vector3f centroid(info.centroid.x(), info.centroid.y(), info.centroid.z());
        const double normalized_distance = static_cast<double>((centroid - context.track_position).norm()) /
                                           static_cast<double>(gate);
        const double temporal_score = clamp01(1.0 - normalized_distance);
        const double combined_score = normalized_shape_weight * clamp01(info.score) +
                                      normalized_temporal_weight * temporal_score;

        if (combined_score > selected_combined_score)
        {
            if (result.found)
            {
                result.second_score = std::max(result.second_score, selected_combined_score);
            }
            result.info = info;
            selected_combined_score = combined_score;
            result.found = true;
        }
        else
        {
            result.second_score = std::max(result.second_score, combined_score);
        }
    }

    if (!result.found)
    {
        return result;
    }

    result.info.is_best = true;
    result.info.score = selected_combined_score;
    for (auto& info : debug_infos)
    {
        if (info.valid && info.cluster_index == result.info.cluster_index)
        {
            info.is_best = true;
            break;
        }
    }

    return result;
}

double CandidateSelector::clamp01(double value)
{
    return std::max(0.0, std::min(1.0, value));
}


}  // namespace rm_radar_lidar_detector
