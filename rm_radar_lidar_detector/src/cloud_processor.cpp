#include "cloud_processor.h"

#include <cmath>
#include <limits>
#include <vector>

namespace rm_radar_lidar_detector
{
    namespace
    {
        constexpr float kHomogeneousCoordW = 1.0f;
    }

    CloudProcessor::PointCloudPtr CloudProcessor::preprocess(const PointCloudPtr& cloud, PlaneRemovalDebug* plane_debug)
    {
        if (plane_debug) {
            plane_debug->reset();
        }

        if (!cloud || cloud->empty()) {
            return PointCloudPtr(new pcl::PointCloud<PointT>);
        }

        // PassThrough filtering
        PointCloudPtr filtered = passthroughFilter(cloud);

        // Voxel downsampling
        PointCloudPtr downsampled = voxelFilter(filtered, params_.voxel_leaf);
        
        // Ground removal
        PointCloudPtr no_ground = removeGround(downsampled, plane_debug);
        
        // Radius outlier removal
        PointCloudPtr result = radiusOutlierRemoval(no_ground);
        // Keep upstream target points if radius filtering becomes too aggressive in sparse frames.
        if (result->empty() && no_ground && !no_ground->empty()) {
            return no_ground;
        }
        
        return result;
    }

    CloudProcessor::PointCloudPtr CloudProcessor::voxelFilter(const PointCloudPtr& cloud, float leaf_size)
    {
        PointCloudPtr result(new pcl::PointCloud<PointT>);
        
        if (!cloud || cloud->empty()) {
            return result;
        }
        
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxel.filter(*result);
        
        return result;
    }

    CloudProcessor::PointCloudPtr CloudProcessor::passthroughFilter(const PointCloudPtr& cloud)
    {
        PointCloudPtr result(new pcl::PointCloud<PointT>);
        
        if (!cloud || cloud->empty()) {
            return result;
        }
        
        // Filter X
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(params_.pass_x_min, params_.pass_x_max);
        pass.filter(*result);
        
        // Filter Y
        pass.setInputCloud(result);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(params_.pass_y_min, params_.pass_y_max);
        pass.filter(*result);
        
        // Filter Z
        pass.setInputCloud(result);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(params_.pass_z_min, params_.pass_z_max);
        pass.filter(*result);
        
        return result;
    }

    CloudProcessor::PointCloudPtr CloudProcessor::removeGround(const PointCloudPtr& cloud, PlaneRemovalDebug* plane_debug)
    {
        if (!cloud || cloud->empty()) {
            return PointCloudPtr(new pcl::PointCloud<PointT>);
        }

        // Unified mode: remove arbitrary planes without splitting classes.
        if (!params_.remove_planes)
        {
            return cloud;
        }

        PointCloudPtr remaining(new pcl::PointCloud<PointT>(*cloud));
        PointCloudPtr removed_planes(new pcl::PointCloud<PointT>);
        removed_planes->reserve(remaining->size() / 2);

        const int plane_min_points = std::max(30, params_.plane_min_points);
        const int max_planes = std::max(1, params_.plane_max_planes);
        const int max_iterations = std::max(20, params_.plane_max_iterations);
        const float ransac_distance = std::max(0.01f, params_.plane_distance_threshold);
        const double min_inlier_ratio =
            std::max(0.0, std::min(1.0, static_cast<double>(params_.plane_min_inlier_ratio)));
        const int fail_streak_threshold = std::max(1, params_.plane_fail_streak_threshold);
        const int fail_cooldown_frames = std::max(0, params_.plane_fail_cooldown_frames);

        const auto register_fail = [&](int& streak, int& cooldown_frames_left) {
            ++streak;
            if (streak >= fail_streak_threshold) {
                cooldown_frames_left = fail_cooldown_frames;
            }
        };
        const auto register_success = [](int& streak, int& cooldown_frames_left) {
            streak = 0;
            cooldown_frames_left = 0;
        };
        const auto append_points = [](const PointCloudPtr& dst, const PointCloudPtr& src) {
            if (!dst || !src || src->empty()) {
                return;
            }
            dst->reserve(dst->size() + src->size());
            *dst += *src;
        };

        if (plane_cooldown_frames_left_ > 0)
        {
            --plane_cooldown_frames_left_;
            return remaining;
        }

        for (int plane_idx = 0;
             plane_idx < max_planes && static_cast<int>(remaining->size()) >= plane_min_points;
             ++plane_idx)
        {
            pcl::SACSegmentation<PointT> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setMaxIterations(max_iterations);
            seg.setDistanceThreshold(ransac_distance);
            seg.setInputCloud(remaining);

            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
            seg.segment(*inliers, *coeffs);
            if (inliers->indices.empty())
            {
                break;
            }

            const int inlier_count = static_cast<int>(inliers->indices.size());
            if (inlier_count < plane_min_points)
            {
                break;
            }
            const double inlier_ratio = static_cast<double>(inlier_count) /
                                        static_cast<double>(remaining->size());
            if (inlier_ratio < min_inlier_ratio)
            {
                break;
            }

            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(remaining);
            extract.setIndices(inliers);

            PointCloudPtr plane_cloud(new pcl::PointCloud<PointT>);
            extract.setNegative(false);
            extract.filter(*plane_cloud);

            PointCloudPtr non_plane_cloud(new pcl::PointCloud<PointT>);
            extract.setNegative(true);
            extract.filter(*non_plane_cloud);

            append_points(removed_planes, plane_cloud);
            remaining.swap(non_plane_cloud);
        }

        const bool accepted = !removed_planes->empty();
        if (accepted)
        {
            if (plane_debug)
            {
                append_points(plane_debug->plane_removed, removed_planes);
            }
            register_success(plane_fail_streak_, plane_cooldown_frames_left_);
        }
        else
        {
            register_fail(plane_fail_streak_, plane_cooldown_frames_left_);
        }

        return remaining;
    }

    CloudProcessor::PointCloudPtr CloudProcessor::radiusOutlierRemoval(const PointCloudPtr& cloud)
    {
        PointCloudPtr result(new pcl::PointCloud<PointT>);
        
        if (!cloud || cloud->empty()) {
            return result;
        }
        
        pcl::RadiusOutlierRemoval<PointT> ror;
        ror.setInputCloud(cloud);
        ror.setRadiusSearch(params_.radius_search);
        ror.setMinNeighborsInRadius(params_.min_neighbors);
        ror.filter(*result);
        
        return result;
    }

    CloudProcessor::PointCloudPtr CloudProcessor::extractInAABB(
        const PointCloudPtr& cloud,
        const pcl::PointXYZ& min_pt,
        const pcl::PointXYZ& max_pt,
        float expansion)
    {
        PointCloudPtr result(new pcl::PointCloud<PointT>);
        
        if (!cloud || cloud->empty()) {
            return result;
        }
        
        Eigen::Vector4f min_point(min_pt.x - expansion, min_pt.y - expansion, 
                                min_pt.z - expansion, kHomogeneousCoordW);
        Eigen::Vector4f max_point(max_pt.x + expansion, max_pt.y + expansion, 
                                max_pt.z + expansion, kHomogeneousCoordW);
        
        pcl::CropBox<PointT> crop_box;
        crop_box.setInputCloud(cloud);
        crop_box.setMin(min_point);
        crop_box.setMax(max_point);
        crop_box.filter(*result);
        
        return result;
    }

    void CloudProcessor::accumulateClouds(
        std::deque<PointCloudPtr>& queue,
        const PointCloudPtr& input,
        PointCloudPtr& output,
        int max_size)
    {
        queue.push_back(input);
        while (static_cast<int>(queue.size()) > max_size) {
            queue.pop_front();
        }
        
        if (!output) {
            output.reset(new pcl::PointCloud<PointT>);
        }
        output->clear();
        size_t total_points = 0;
        for (const auto& cloud : queue) {
            if (cloud) {
                total_points += cloud->size();
            }
        }
        output->reserve(total_points);
        
        for (const auto& cloud : queue) {
            if (cloud && !cloud->empty()) {
                *output += *cloud;
            }
        }
    }

}  // namespace rm_radar_lidar_detector
