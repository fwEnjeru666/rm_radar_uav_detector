#include "dbscan.h"

#include <algorithm>
#include <cmath>

namespace rm_radar_lidar_detector
{

    DBSCAN::DBSCAN(std::vector<ClusterPoint>& points, double eps, unsigned int minPts)
        : points_(points), eps_(eps), minPts_(minPts), cluster_id_(0), octree_(eps)
    {
        buildOctree();
    }

    void DBSCAN::buildOctree()
    {
        cloud_.reset(new pcl::PointCloud<PointT>);
        cloud_->reserve(points_.size());
        
        for (const auto& p : points_) {
            cloud_->push_back(PointT(p.x, p.y, p.z));
        }
        
        if (!cloud_->empty()) {
            octree_.setInputCloud(cloud_);
            octree_.addPointsFromInputCloud();
        }
    }

    double DBSCAN::getDistance(const ClusterPoint& p1, const ClusterPoint& p2)
    {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    std::vector<int> DBSCAN::regionQuery(int point_idx)
    {
        std::vector<int> neighbors;
        
        if (cloud_->empty() || point_idx >= static_cast<int>(cloud_->size())) {
            return neighbors;
        }
        
        PointT search_point = cloud_->points[point_idx];
        std::vector<int> indices;
        std::vector<float> distances;
        
        octree_.radiusSearch(search_point, eps_, indices, distances);
        
        return indices;
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
        }
        
        // Remove point_idx from seeds
        seeds.erase(std::remove(seeds.begin(), seeds.end(), point_idx), seeds.end());
        
        // Process seeds
        while (!seeds.empty()) {
            int current_point = seeds.back();
            seeds.pop_back();
            
            std::vector<int> result = regionQuery(current_point);
            
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
            if (points_[i].visited) {
                continue;
            }
            
            points_[i].visited = true;
            std::vector<int> neighbors = regionQuery(i);
            
            if (neighbors.size() < minPts_) {
                points_[i].cluster_id = -1;  // Noise
            } else {
                cluster_id_++;
                expandCluster(i, cluster_id_);
            }
        }
        
        // Build cluster point clouds
        clusters_.clear();
        for (size_t i = 0; i < points_.size(); ++i) {
            int cid = points_[i].cluster_id;
            if (cid > 0) {
                if (clusters_.find(cid) == clusters_.end()) {
                    clusters_[cid].reset(new pcl::PointCloud<pcl::PointXYZ>);
                }
                clusters_[cid]->push_back(pcl::PointXYZ(points_[i].x, points_[i].y, points_[i].z));
            }
        }
        
        return cluster_id_;
    }

}  // namespace rm_radar_lidar_detector
