#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

#include <Eigen/Dense>
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
    double min_pca_r = 1.0;
    double max_pca_r = 10.0;
    float min_h = -3.0f;
    float max_h = 10.0f;
};

struct ClusterDebugInfo {
    int cluster_index = -1;
    bool valid = false;
    bool is_best = false;
    int point_count = 0;
    double volume = 0.0;
    double ratio = 0.0;
    double pca_ratio = 0.0;
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

    /**
     * @brief Compute PCA axis ratio sqrt(lambda_max/lambda_min)
     */
    static double getPcaRatio(const PointCloudPtr& cluster);

private:
    bool validateMetrics(int point_count, double volume, double ratio, double pca_ratio, float center_z) const;
    ClusterFilterParams params_;
};

struct CandidateSelectionContext
{
    bool use_track_position = false;
    Eigen::Vector3f track_position = Eigen::Vector3f::Zero();
    float track_gate_distance = 0.05f;
    double shape_weight = 0.35;
    double temporal_weight = 0.65;
};

struct CandidateSelectionResult
{
    bool found = false;
    ClusterDebugInfo info;
    double second_score = -1.0;
    int valid_count = 0;
};

class CandidateSelector
{
public:
    using ClusterMap = ClusterFilter::ClusterMap;

    CandidateSelectionResult select(const ClusterMap& clusters,
                                    std::vector<ClusterDebugInfo>& debug_infos,
                                    const CandidateSelectionContext& context) const;

private:
    CandidateSelectionResult selectByShape(std::vector<ClusterDebugInfo>& debug_infos) const;
    CandidateSelectionResult selectByTrackPosition(const ClusterMap& clusters,
                                                   std::vector<ClusterDebugInfo>& debug_infos,
                                                   const CandidateSelectionContext& context) const;
    static double clamp01(double value);
};

}  // namespace rm_radar_lidar_detector
