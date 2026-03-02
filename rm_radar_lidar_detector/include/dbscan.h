#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_search.h>

#include <vector>
#include <unordered_map>

#include "types.h"

namespace rm_radar_lidar_detector
{

class DBSCAN {
public:
    using PointT = pcl::PointXYZ;
    using PointCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

    /**
     * @brief Constructor
     * @param points Reference to points for clustering
     * @param eps Epsilon - neighborhood radius
     * @param minPts Minimum points to form a core point
     */
    DBSCAN(std::vector<ClusterPoint>& points, double eps, unsigned int minPts);
    ~DBSCAN() = default;

    /**
     * @brief Run DBSCAN clustering
     * @return Number of clusters found
     */
    int run();
    
    /**
     * @brief Find all neighbors within epsilon distance
     * @param point_idx Index of the query point
     * @return Vector of neighbor indices
     */
    std::vector<int> regionQuery(int point_idx);
    
    /**
     * @brief Expand cluster from a core point
     * @param point_idx Index of the core point
     * @param cluster_id ID to assign to the cluster
     * @return cluster_id on success, -1 if point is noise
     */
    int expandCluster(int point_idx, int cluster_id);
    
    /**
     * @brief Calculate Euclidean distance between two points
     */
    double getDistance(const ClusterPoint& p1, const ClusterPoint& p2);
    
    /**
     * @brief Get number of clusters found
     */
    int getClusterCount() const { return cluster_id_; }
    
    /**
     * @brief Get epsilon value
     */
    double getEpsilon() const { return eps_; }
    
    /**
     * @brief Get clustered point clouds
     * @return Map of cluster_id to point cloud
     */
    const std::unordered_map<int, PointCloudPtr>& getClusters() const { 
        return clusters_; 
    }

    std::vector<ClusterPoint>& points_;

private:
    /**
     * @brief Build octree from points for efficient neighbor search
     */
    void buildOctree();
    
    double eps_;
    unsigned int minPts_;
    int cluster_id_;
    
    pcl::PointCloud<PointT>::Ptr cloud_;
    pcl::octree::OctreePointCloudSearch<PointT> octree_;
    std::unordered_map<int, PointCloudPtr> clusters_;
};

}  // namespace rm_radar_lidar_detector
