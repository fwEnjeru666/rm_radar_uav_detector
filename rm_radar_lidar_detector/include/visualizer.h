#pragma once

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
#include <string>

#include "types.h"

namespace rm_radar_lidar_detector
{

class Visualizer {
    public:
        using PointT = pcl::PointXYZ;
        using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
        
        Visualizer() = default;
        ~Visualizer() = default;
        
        void init(ros::NodeHandle& nh, const std::string& prefix = "");
        
        void publishCloud(const PointCloudPtr& cloud,
                        const std::string& frame_id,
                        const ros::Time& stamp,
                        ros::Publisher& pub);
        
        static visualization_msgs::Marker createAABBMarker(
            const pcl::PointXYZ& min_pt,
            const pcl::PointXYZ& max_pt,
            int id,
            const std::string& ns,
            const std::string& frame_id,
            const Eigen::Vector4d& color,
            double lifetime = 0.2);
        
        static visualization_msgs::Marker createAABBMarker(
            const BBox3D& bbox,
            int id,
            const std::string& ns,
            const std::string& frame_id,
            const Eigen::Vector4d& color,
            double lifetime = 0.2);
        
        static visualization_msgs::Marker createOBBMarker(
            const PointCloudPtr& cluster,
            int id,
            const std::string& frame_id,
            const Eigen::Vector4d& color,
            double lifetime = 0.2);
        
        static visualization_msgs::Marker createCentroidMarker(
            const Eigen::Vector4f& centroid,
            int id,
            const std::string& frame_id,
            const Eigen::Vector4d& color,
            double scale = 0.2,
            double lifetime = 0.2);
        
        static visualization_msgs::Marker createTextMarker(
            const Eigen::Vector3f& position,
            const std::string& text,
            int id,
            const std::string& frame_id,
            double scale = 0.3,
            double lifetime = 0.2);
        
        static void broadcastTF(const Eigen::Vector3f& position,
                                const Eigen::Quaternionf& orientation,
                                const std::string& parent_frame,
                                const std::string& child_frame,
                                tf::TransformBroadcaster& broadcaster);

    private:
        ros::Publisher marker_pub_;
        ros::Publisher marker_array_pub_;
    };

}  // namespace rm_radar_lidar_detector
