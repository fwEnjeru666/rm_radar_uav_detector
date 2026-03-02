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
        if (point_count < params_.min_points || point_count > params_.max_points) {
            return false;
        }
        
        // Check volume
        double volume = getVolume(min_pt, max_pt);
        if (volume < params_.min_volume || volume > params_.max_volume) {
            return false;
        }
        
        // Check ratio
        double ratio = getRatio(min_pt, max_pt);
        if (ratio < params_.min_ratio || ratio > params_.max_ratio) {
            return false;
        }
        
        // Check center Z
        float center_z = (min_pt.z + max_pt.z) / 2.0f;
        if (center_z < params_.min_z) {
            return false;
        }
        
        return true;
    }

    ClusterFilter::PointCloudPtr ClusterFilter::findBest(
        const ClusterMap& clusters,
        BBox3D& best_bbox,
        Eigen::Vector4f& best_centroid)
    {
        PointCloudPtr best_cluster = nullptr;
        size_t max_points = 0;
        
        for (const auto& pair : clusters) {
            auto cluster = pair.second;
            
            if (!cluster || cluster->empty()) {
                continue;
            }
            
            // Compute bounding box
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);
            
            // Validate cluster
            if (!validate(cluster, min_pt, max_pt)) {
                continue;
            }
            
            // Select largest valid cluster
            if (cluster->size() > max_points) {
                max_points = cluster->size();
                best_cluster = cluster;
                best_bbox.update(min_pt, max_pt);
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
