#include "dynamic_detector.h"

namespace rm_radar_lidar_detector
{

    std::vector<ClusterPoint> DynamicDetector::detect(
        const PointCloudPtr& cur_cloud,
        const PointCloudPtr& prev_cloud)
    {
        std::vector<ClusterPoint> dynamic_points;
        
        if (!cur_cloud || !prev_cloud || cur_cloud->empty() || prev_cloud->empty()) {
            last_detection_count_ = 0;
            return dynamic_points;
        }
        
        // Build KdTree for previous cloud
        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(prev_cloud);
        
        // Find dynamic points (points in current cloud not present in previous)
        for (const auto& pt : cur_cloud->points) {
            std::vector<int> indices;
            std::vector<float> distances;
            
            if (kdtree.radiusSearch(pt, params_.distance_threshold, indices, distances) == 0) {
                // Point not found in previous cloud - dynamic point
                // Filter by minimum Z
                if (pt.z >= params_.min_z) {
                    dynamic_points.emplace_back(pt.x, pt.y, pt.z);
                }
            }
        }
        
        last_detection_count_ = dynamic_points.size();
        return dynamic_points;
    }

    std::vector<ClusterPoint> DynamicDetector::detectWithOctree(
        const PointCloudPtr& cur_cloud,
        const PointCloudPtr& prev_cloud,
        float resolution)
    {
        std::vector<ClusterPoint> dynamic_points;
        
        if (!cur_cloud || !prev_cloud || cur_cloud->empty() || prev_cloud->empty()) {
            last_detection_count_ = 0;
            return dynamic_points;
        }
        
        // Build octree for previous cloud
        pcl::octree::OctreePointCloudSearch<PointT> octree(resolution);
        octree.setInputCloud(prev_cloud);
        octree.addPointsFromInputCloud();
        
        // Find dynamic points
        for (const auto& pt : cur_cloud->points) {
            std::vector<int> indices;
            std::vector<float> distances;
            
            if (octree.radiusSearch(pt, params_.distance_threshold, indices, distances) == 0) {
                if (pt.z >= params_.min_z) {
                    dynamic_points.emplace_back(pt.x, pt.y, pt.z);
                }
            }
        }
        
        last_detection_count_ = dynamic_points.size();
        return dynamic_points;
    }

}  // namespace rm_radar_lidar_detector
