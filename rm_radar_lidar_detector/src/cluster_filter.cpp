#include "cluster_filter.h"

#include <algorithm>
#include <cmath>

namespace rm_radar_lidar_detector
{

    bool ClusterFilter::validate(
        const PointCloudPtr& cluster,
        const pcl::PointXYZ& min_pt,
        const pcl::PointXYZ& max_pt)
    {
        if (!cluster || cluster->empty()) {
            return false;
        }
        
        // Check point count
        int point_count = static_cast<int>(cluster->size());
        if (point_count < params_.min_n || point_count > params_.max_n) {
            return false;
        }
        
        // Check volume
        double volume = getVolume(min_pt, max_pt);
        if (volume < params_.min_v || volume > params_.max_v) {
            return false;
        }
        
        // Check ratio
        double ratio = getRatio(min_pt, max_pt);
        if (ratio < params_.min_r || ratio > params_.max_r) {
            return false;
        }
        
        // Check center Z
        float center_z = (min_pt.z + max_pt.z) / 2.0f;
        const float min_h = std::min(params_.min_h, params_.max_h);
        const float max_h = std::max(params_.min_h, params_.max_h);
        if (center_z < min_h || center_z > max_h) {
            return false;
        }
        
        return true;
    }

    ClusterFilter::PointCloudPtr ClusterFilter::findBest(
        const ClusterMap& clusters,
        BBox3D& best_bbox,
        Eigen::Vector4f& best_centroid,
        std::vector<ClusterDebugInfo>* debug_infos)
    {
        PointCloudPtr best_cluster = nullptr;
        double best_score = -1.0;
        int best_cluster_index = -1;

        if (debug_infos) {
            debug_infos->clear();
            debug_infos->reserve(clusters.size());
        }

        auto clamp01 = [](double v) {
            return std::max(0.0, std::min(1.0, v));
        };
        
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

            const double point_count = static_cast<double>(info.point_count);
            const double ratio = info.ratio;
            const double volume = info.volume;
            const double center_z = static_cast<double>(min_pt.z + max_pt.z) * 0.5;

            const double point_den = std::max(1.0, static_cast<double>(params_.max_n - params_.min_n));
            const double point_score = clamp01((point_count - static_cast<double>(params_.min_n)) / point_den);

            const double ratio_mid = 0.5 * (params_.min_r + params_.max_r);
            const double ratio_half = std::max(1e-6, 0.5 * (params_.max_r - params_.min_r));
            const double ratio_score = clamp01(1.0 - std::abs(ratio - ratio_mid) / ratio_half);

            const double vol_mid = 0.5 * (params_.min_v + params_.max_v);
            const double vol_half = std::max(1e-6, 0.5 * (params_.max_v - params_.min_v));
            const double volume_score = clamp01(1.0 - std::abs(volume - vol_mid) / vol_half);

            const double height_norm = std::max(1e-6, params_.score_height_norm_span);
            const double height_score = clamp01((center_z - static_cast<double>(params_.min_h)) / height_norm);

            const double w_points = std::max(0.0, params_.score_weight_points);
            const double w_ratio = std::max(0.0, params_.score_weight_ratio);
            const double w_volume = std::max(0.0, params_.score_weight_volume);
            const double w_height = std::max(0.0, params_.score_weight_height);
            const double w_sum = std::max(1e-9, w_points + w_ratio + w_volume + w_height);

            const double score =
                (w_points * point_score +
                 w_ratio * ratio_score +
                 w_volume * volume_score +
                 w_height * height_score) / w_sum;
            info.score = score;
            
            // Validate cluster
            if (!validate(cluster, min_pt, max_pt)) {
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
        
        if (best_cluster && !best_cluster->empty()) {
            best_centroid = getCentroid(best_cluster);
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
        
        if (dz < 0.001) {
            return 100.0;  // Invalid ratio for flat objects
        }
        double ratio = (dx * dy) / dz;
        return ratio;
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

}  // namespace rm_radar_lidar_detector
