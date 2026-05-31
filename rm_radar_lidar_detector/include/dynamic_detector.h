#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>

#include <vector>

#include "types.h"

namespace rm_radar_lidar_detector
{

struct DynamicDetectorParams {
    float distance_threshold = 0.3f;
    float min_z = -14.0f;
};

class DynamicDetector {
public:
    using PointT = pcl::PointXYZ;
    using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
    
    DynamicDetector() = default;
    ~DynamicDetector() = default;
    
    void setParams(const DynamicDetectorParams& params) { params_ = params; }
    const DynamicDetectorParams& getParams() const { return params_; }
    
    std::vector<ClusterPoint> detect(const PointCloudPtr& cur_cloud,
                                     const PointCloudPtr& prev_cloud);
    
    std::vector<ClusterPoint> detectWithOctree(const PointCloudPtr& cur_cloud,
                                               const PointCloudPtr& prev_cloud,
                                               float resolution);
    
    size_t getLastDetectionCount() const { return last_detection_count_; }

private:
    DynamicDetectorParams params_;
    size_t last_detection_count_ = 0;
    std::vector<int> nn_indices_buffer_;
    std::vector<float> nn_distances_buffer_;
};

}  // namespace rm_radar_lidar_detector
