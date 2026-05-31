#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <deque>
#include <vector>

#include "types.h"

namespace rm_radar_lidar_detector
{

struct FilterParams {
    float pass_x_min = 0.0f;
    float pass_x_max = 35.0f;
    float pass_y_min = -10.0f;
    float pass_y_max = 8.0f;
    float pass_z_min = -50.0f;
    float pass_z_max = 7.0f;
    
    float voxel_leaf = 0.05f;
    
    float radius_search = 0.3f;
    int min_neighbors = 3;
    
    // Unified plane-removal controls.
    bool remove_planes = true;
    float plane_distance_threshold = 0.3f;
    int plane_max_iterations = 120;
    int plane_max_planes = 3;
    int plane_min_points = 120;
    float plane_min_inlier_ratio = 0.08f;
    int plane_fail_streak_threshold = 2;
    int plane_fail_cooldown_frames = 20;
};

class CloudProcessor {
public:
    using PointT = pcl::PointXYZ;
    using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;

    struct PlaneRemovalDebug {
        PointCloudPtr plane_removed;

        PlaneRemovalDebug()
            : plane_removed(new pcl::PointCloud<PointT>) {}

        void reset() {
            plane_removed->clear();
        }
    };
    
    CloudProcessor() = default;
    ~CloudProcessor() = default;
    
    void setParams(const FilterParams& params) { params_ = params; }
    const FilterParams& getParams() const { return params_; }
    
    PointCloudPtr preprocess(const PointCloudPtr& cloud, PlaneRemovalDebug* plane_debug = nullptr);
    PointCloudPtr voxelFilter(const PointCloudPtr& cloud, float leaf_size);
    PointCloudPtr passthroughFilter(const PointCloudPtr& cloud);
    PointCloudPtr removeGround(const PointCloudPtr& cloud, PlaneRemovalDebug* plane_debug = nullptr);
    PointCloudPtr radiusOutlierRemoval(const PointCloudPtr& cloud);
    
    PointCloudPtr extractInAABB(const PointCloudPtr& cloud,
                                const pcl::PointXYZ& min_pt,
                                const pcl::PointXYZ& max_pt,
                                float expansion = 0.0f);
    
    static void accumulateClouds(std::deque<PointCloudPtr>& queue,
                                 const PointCloudPtr& input,
                                 PointCloudPtr& output,
                                 int max_size);

private:
    FilterParams params_;
    // Runtime state: avoid repeated stripping attempts when recent frames fail acceptance checks.
    int plane_fail_streak_ = 0;
    int plane_cooldown_frames_left_ = 0;
};

}  // namespace rm_radar_lidar_detector
