#pragma once

#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <vector>

namespace rm_radar_lidar_detector
{

    struct ClusterPoint {
        float x, y, z;
        std::vector<float> coords;
        bool visited = false;
        int cluster_id = -1;  // -1: noise, 0: unclassified, 1..n: cluster id
        
        ClusterPoint(float x, float y, float z) : x(x), y(y), z(z) {
            coords = {x, y, z};
        }
    };


    struct BBox3D {
        Eigen::Vector3f min_pt;
        Eigen::Vector3f max_pt;
        Eigen::Vector3f center;
        Eigen::Vector3f dimensions;
        Eigen::Quaternionf orientation;
        bool valid = false;
        
        BBox3D() : min_pt(Eigen::Vector3f::Zero()), 
                max_pt(Eigen::Vector3f::Zero()),
                center(Eigen::Vector3f::Zero()),
                dimensions(Eigen::Vector3f::Zero()),
                orientation(Eigen::Quaternionf::Identity()),
                valid(false) {}
        
        float volume() const {
            return dimensions.x() * dimensions.y() * dimensions.z();
        }
        
        float ratio() const {
            float max_dim = dimensions.maxCoeff();
            float min_dim = dimensions.minCoeff();
            return (min_dim > 0.001f) ? max_dim / min_dim : 0.0f;
        }
        
        void update(const pcl::PointXYZ& min, const pcl::PointXYZ& max) {
            min_pt = Eigen::Vector3f(min.x, min.y, min.z);
            max_pt = Eigen::Vector3f(max.x, max.y, max.z);
            center = (min_pt + max_pt) / 2.0f;
            dimensions = max_pt - min_pt;
            valid = true;
        }
        
        void reset() {
            min_pt = Eigen::Vector3f::Zero();
            max_pt = Eigen::Vector3f::Zero();
            center = Eigen::Vector3f::Zero();
            dimensions = Eigen::Vector3f::Zero();
            valid = false;
        }
    };


    struct BBox2D {
        Eigen::Vector2f uv_min;
        Eigen::Vector2f uv_max;
        Eigen::Vector2f centroid;
        
        BBox2D() : uv_min(Eigen::Vector2f::Zero()),
                uv_max(Eigen::Vector2f::Zero()),
                centroid(Eigen::Vector2f::Zero()) {}
    };


    enum class TrackState {
        LOST = 0,
        TRACKING = 1,
        SEARCHING = 2
    };

}  // namespace rm_radar_lidar_detector
