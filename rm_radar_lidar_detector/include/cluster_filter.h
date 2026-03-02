#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

#include <unordered_map>

#include "types.h"

namespace rm_radar_lidar_detector
{

struct ClusterFilterParams {
    double max_volume = 10.0;
    double min_volume = 0.1;     // compatible default
    double max_ratio = 10.0;
    double min_ratio = 0.1;      // compatible default
    float min_z = -3.0f;         // compatible: aircraft_min_z default
    int min_points = 5;
    int max_points = 5000;       // Increased for accumulated cloud
};

/**
 * @brief Cluster filter - validates and selects best clusters
 */
class ClusterFilter {
public:
    using PointT = pcl::PointXYZ;
    using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
    using ClusterMap = std::unordered_map<int, PointCloudPtr>;
    
    ClusterFilter() = default;
    ~ClusterFilter() = default;
    
    /**
     * @brief Set filter parameters
     */
    void setParams(const ClusterFilterParams& params) { params_ = params; }
    
    /**
     * @brief Get current parameters
     */
    const ClusterFilterParams& getParams() const { return params_; }
    
    /**
     * @brief Validate a single cluster
     * @param cluster Point cloud cluster
     * @param min_pt Minimum point of AABB
     * @param max_pt Maximum point of AABB
     * @return true if cluster passes all filters
     */
    bool validate(const PointCloudPtr& cluster,
                  const pcl::PointXYZ& min_pt,
                  const pcl::PointXYZ& max_pt);
    
    /**
     * @brief Find the best cluster from a map of clusters
     * @param clusters Map of cluster_id to point cloud
     * @param best_bbox Output: bounding box of best cluster
     * @param best_centroid Output: centroid of best cluster
     * @return Best cluster point cloud, nullptr if none valid
     */
    PointCloudPtr findBest(const ClusterMap& clusters,
                           BBox3D& best_bbox,
                           Eigen::Vector4f& best_centroid);
    
    /**
     * @brief Compute volume of AABB
     */
    static double getVolume(const pcl::PointXYZ& min_pt, const pcl::PointXYZ& max_pt);
    
    /**
     * @brief Compute dimension ratio (max/min)
     */
    static double getRatio(const pcl::PointXYZ& min_pt, const pcl::PointXYZ& max_pt);
    
    /**
     * @brief Compute centroid of a cluster
     */
    static Eigen::Vector4f getCentroid(const PointCloudPtr& cluster);

private:
    ClusterFilterParams params_;
};

}  // namespace rm_radar_lidar_detector
