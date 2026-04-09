#include "dbscan.h"

#include <algorithm>
#include <boost/make_shared.hpp>

namespace rm_radar_lidar_detector
{

    DBSCAN::DBSCAN(std::vector<ClusterPoint>& points, double eps, unsigned int minPts)
        : points_(points), eps_(eps), minPts_(minPts), cluster_id_(0), octree_(eps)
    {
        buildOctree();
    }

    void DBSCAN::buildOctree()
    {
        cloud_ = boost::make_shared<pcl::PointCloud<PointT>>();
        cloud_->reserve(points_.size());
        
        for (const auto& p : points_) {
            cloud_->push_back(PointT(p.x, p.y, p.z));
        }
        
        if (!cloud_->empty()) {
            octree_.setInputCloud(cloud_);
            octree_.addPointsFromInputCloud();
        }
    }

    const std::vector<int>& DBSCAN::regionQuery(int point_idx)
    {
        nn_distances_buffer_.clear();
        nn_neighbors_buffer_.clear();
        
        if (cloud_->empty() || point_idx >= static_cast<int>(cloud_->size())) {
            return nn_neighbors_buffer_;
        }
        
        PointT search_point = cloud_->points[point_idx];

        octree_.radiusSearch(search_point, eps_, nn_neighbors_buffer_, nn_distances_buffer_);
        return nn_neighbors_buffer_;
    }

    int DBSCAN::expandCluster(int point_idx, int cluster_id)
    {
        std::vector<int> seeds = regionQuery(point_idx);
        
        if (seeds.size() < minPts_) {
            points_[point_idx].cluster_id = -1;  // Mark as noise
            return -1;
        }
        
        // All points in seeds are density-reachable from point_idx
        for (int seed_idx : seeds) {
            points_[seed_idx].cluster_id = cluster_id;
            points_[seed_idx].visited = true;
        }
        
        // Process seeds
        while (!seeds.empty()) {
            int current_point = seeds.back();
            seeds.pop_back();

            if (current_point == point_idx) {
                continue;
            }
            
            const std::vector<int>& result = regionQuery(current_point);
            
            if (result.size() >= minPts_) {
                for (int result_idx : result) {
                    if (points_[result_idx].cluster_id == 0) {  // Unclassified
                        seeds.push_back(result_idx);
                        points_[result_idx].cluster_id = cluster_id;
                    } else if (points_[result_idx].cluster_id == -1) {  // Was noise
                        points_[result_idx].cluster_id = cluster_id;
                    }
                }
            }
        }
        
        return cluster_id;
    }

    int DBSCAN::run()
    {
        if (points_.empty()) {
            return 0;
        }
        
        // Initialize all points as unclassified
        for (auto& p : points_) {
            p.cluster_id = 0;
            p.visited = false;
        }
        
        cluster_id_ = 0;
        for (size_t i = 0; i < points_.size(); ++i) {
            if (points_[i].cluster_id != 0) {
                continue;
            }
            
            points_[i].visited = true;
            const std::vector<int>& neighbors = regionQuery(static_cast<int>(i));
            
            if (neighbors.size() < minPts_) {
                points_[i].cluster_id = -1;  // Noise
            } else {
                cluster_id_++;
                expandCluster(i, cluster_id_);
            }
        }
        
        // Build cluster point clouds
        clusters_.clear();
        std::vector<size_t> cluster_sizes(static_cast<size_t>(cluster_id_ + 1), 0);

        for (size_t i = 0; i < points_.size(); ++i) {
            int cid = points_[i].cluster_id;
            if (cid > 0) {
                ++cluster_sizes[static_cast<size_t>(cid)];
            }
        }

        clusters_.resize(static_cast<size_t>(cluster_id_ + 1));
        for (int cid = 1; cid <= cluster_id_; ++cid) {
            auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            cloud->reserve(cluster_sizes[static_cast<size_t>(cid)]);
            clusters_[static_cast<size_t>(cid)] = cloud;
        }

        for (size_t i = 0; i < points_.size(); ++i) {
            int cid = points_[i].cluster_id;
            if (cid > 0) {
                clusters_[static_cast<size_t>(cid)]->push_back(
                    pcl::PointXYZ(points_[i].x, points_[i].y, points_[i].z));
            }
        }
        
        return cluster_id_;
    }

}  // namespace rm_radar_lidar_detector
