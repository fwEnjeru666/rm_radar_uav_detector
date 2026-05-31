#pragma once

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Point.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <deque>

#include "types.h"
#include "cluster_filter.h"
#include "track.h"

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

        static visualization_msgs::MarkerArray createClusterDebugMarkerArray(
            const std::vector<ClusterDebugInfo>& debug_infos,
            const std::string& frame_id,
            const ros::Time& stamp,
            double box_lifetime = 0.25,
            double text_lifetime = 0.25);

        static visualization_msgs::Marker createDeleteMarker(
            const std::string& frame_id,
            const std::string& ns,
            int id,
            const ros::Time& stamp);

        static visualization_msgs::Marker createTrackTargetSphereMarker(
            const SingleTargetTracker::State& state,
            bool has_track,
            const std::string& frame_id,
            int id,
            const ros::Time& stamp,
            double lifetime,
            double sphere_scale);

        static visualization_msgs::Marker createTrackVelocityArrowMarker(
            const SingleTargetTracker::State& state,
            bool has_track,
            const std::string& frame_id,
            int id,
            const ros::Time& stamp,
            double lifetime,
            double shaft_diameter,
            double head_diameter,
            double head_length,
            double min_speed,
            double length_scale,
            double max_length);

        static visualization_msgs::Marker createTrackTrajectoryMarker(
            const SingleTargetTracker::State& state,
            bool has_track,
            bool publish_trajectory,
            const std::string& frame_id,
            int id,
            const ros::Time& stamp,
            double line_width,
            float min_step,
            size_t max_points,
            std::deque<geometry_msgs::Point>& trajectory_points);

        static visualization_msgs::Marker createTrackSpeedTextMarker(
            const SingleTargetTracker::State& state,
            bool has_track,
            const std::string& frame_id,
            int id,
            const ros::Time& stamp,
            double lifetime,
            float z_offset,
            float text_scale);
        
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
