#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

#include <vector>

#include "types.h"

namespace rm_radar_lidar_detector
{

struct ClusterFilterParams {
    int min_n = 5;
    int max_n = 5000;            // Increased for accumulated cloud
    double min_v = 0.1;
    double max_v = 10.0;
    double min_r = 0.1;
    double max_r = 10.0;
    float min_h = -3.0f;
    float max_h = 10.0f;
    double score_weight_points = 0.45;
    double score_weight_ratio = 0.20;
    double score_weight_volume = 0.20;
    double score_weight_height = 0.15;
    double score_height_norm_span = 5.0;
};

struct ClusterDebugInfo {
    int cluster_index = -1;
    bool valid = false;
    bool is_best = false;
    int point_count = 0;
    double volume = 0.0;
    double ratio = 0.0;
    double score = 0.0;
    Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
    BBox3D bbox;
};

/**
 * @brief Cluster filter - validates and selects best clusters
 */
class ClusterFilter {
public:
    using PointT = pcl::PointXYZ;
    using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
    using ClusterMap = std::vector<PointCloudPtr>;
    
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
    * @brief Find the best cluster from a vector of clusters
    * @param clusters Vector indexed by cluster_id (index 0 may be empty)
     * @param best_bbox Output: bounding box of best cluster
     * @param best_centroid Output: centroid of best cluster
     * @return Best cluster point cloud, nullptr if none valid
     */
    PointCloudPtr findBest(const ClusterMap& clusters,
                           BBox3D& best_bbox,
                           Eigen::Vector4f& best_centroid,
                           std::vector<ClusterDebugInfo>* debug_infos = nullptr);
    
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
