#include "cloud_processor.h"

namespace rm_radar_lidar_detector
{

    CloudProcessor::PointCloudPtr CloudProcessor::preprocess(const PointCloudPtr& cloud)
    {
        if (!cloud || cloud->empty()) {
            return PointCloudPtr(new pcl::PointCloud<PointT>);
        }

        // PassThrough filtering
        PointCloudPtr filtered = passthroughFilter(cloud);

        // Voxel downsampling
        PointCloudPtr downsampled = voxelFilter(filtered, params_.voxel_leaf);
        
        // Ground removal
        PointCloudPtr no_ground = removeGround(downsampled);
        
        // Radius outlier removal
        PointCloudPtr result = radiusOutlierRemoval(no_ground);
        
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
        
        PointCloudPtr temp = cloud;
        
        // Filter X
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(temp);
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

    CloudProcessor::PointCloudPtr CloudProcessor::removeGround(const PointCloudPtr& cloud)
    {
        PointCloudPtr result(new pcl::PointCloud<PointT>);
        
        if (!cloud || cloud->empty()) {
            return result;
        }
        
        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients(false);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(params_.ransac_distance_threshold);
        seg.setInputCloud(cloud);
        
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        seg.segment(*inliers, *coefficients);
        
        if (inliers->indices.empty()) {
            *result = *cloud;
            return result;
        }
        
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*result);
        
        return result;
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
                                min_pt.z - expansion, 1.0f);
        Eigen::Vector4f max_point(max_pt.x + expansion, max_pt.y + expansion, 
                                max_pt.z + expansion, 1.0f);
        
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
        
        for (const auto& cloud : queue) {
            *output += *cloud;
        }
    }

}  // namespace rm_radar_lidar_detector
