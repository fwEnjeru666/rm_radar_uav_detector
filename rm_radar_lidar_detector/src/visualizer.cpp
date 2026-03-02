#include "visualizer.h"

namespace rm_radar_lidar_detector
{
    void Visualizer::init(ros::NodeHandle& nh, const std::string& prefix)
    {
        std::string topic_prefix = prefix.empty() ? "" : prefix + "/";
        marker_pub_ = nh.advertise<visualization_msgs::Marker>(topic_prefix + "marker", 10);
        marker_array_pub_ = nh.advertise<visualization_msgs::MarkerArray>(topic_prefix + "markers", 10);
    }

    void Visualizer::publishCloud(
        const PointCloudPtr& cloud,
        const std::string& frame_id,
        const ros::Time& stamp,
        ros::Publisher& pub)
    {
        if (!cloud || cloud->empty()) {
            return;
        }
        
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.frame_id = frame_id;
        msg.header.stamp = stamp;
        pub.publish(msg);
    }

    visualization_msgs::Marker Visualizer::createAABBMarker(
        const pcl::PointXYZ& min_pt,
        const pcl::PointXYZ& max_pt,
        int id,
        const std::string& ns,
        const std::string& frame_id,
        const Eigen::Vector4d& color,
        double lifetime)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = ns;
        marker.id = id;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        
        marker.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
        marker.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
        marker.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = std::max(0.1f, max_pt.x - min_pt.x);
        marker.scale.y = std::max(0.1f, max_pt.y - min_pt.y);
        marker.scale.z = std::max(0.1f, max_pt.z - min_pt.z);
        
        marker.color.r = color(0);
        marker.color.g = color(1);
        marker.color.b = color(2);
        marker.color.a = color(3);
        
        marker.lifetime = ros::Duration(lifetime);
        
        return marker;
    }

    visualization_msgs::Marker Visualizer::createAABBMarker(
        const BBox3D& bbox,
        int id,
        const std::string& ns,
        const std::string& frame_id,
        const Eigen::Vector4d& color,
        double lifetime)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = ns;
        marker.id = id;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        
        marker.pose.position.x = bbox.center.x();
        marker.pose.position.y = bbox.center.y();
        marker.pose.position.z = bbox.center.z();
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = std::max(0.1f, bbox.dimensions.x());
        marker.scale.y = std::max(0.1f, bbox.dimensions.y());
        marker.scale.z = std::max(0.1f, bbox.dimensions.z());
        
        marker.color.r = color(0);
        marker.color.g = color(1);
        marker.color.b = color(2);
        marker.color.a = color(3);
        
        marker.lifetime = ros::Duration(lifetime);
        
        return marker;
    }

    visualization_msgs::Marker Visualizer::createOBBMarker(
        const PointCloudPtr& cluster,
        int id,
        const std::string& frame_id,
        const Eigen::Vector4d& color,
        double lifetime)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "obb";
        marker.id = id;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        
        if (!cluster || cluster->empty()) {
            marker.action = visualization_msgs::Marker::DELETE;
            return marker;
        }
        
        pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
        feature_extractor.setInputCloud(cluster);
        feature_extractor.compute();
        
        pcl::PointXYZ min_point_OBB, max_point_OBB, position_OBB;
        Eigen::Matrix3f rotational_matrix_OBB;
        
        feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
        
        Eigen::Quaternionf quat(rotational_matrix_OBB);
        
        marker.pose.position.x = position_OBB.x;
        marker.pose.position.y = position_OBB.y;
        marker.pose.position.z = position_OBB.z;
        marker.pose.orientation.x = quat.x();
        marker.pose.orientation.y = quat.y();
        marker.pose.orientation.z = quat.z();
        marker.pose.orientation.w = quat.w();
        
        marker.scale.x = std::max(0.1f, max_point_OBB.x - min_point_OBB.x);
        marker.scale.y = std::max(0.1f, max_point_OBB.y - min_point_OBB.y);
        marker.scale.z = std::max(0.1f, max_point_OBB.z - min_point_OBB.z);
        
        marker.color.r = color(0);
        marker.color.g = color(1);
        marker.color.b = color(2);
        marker.color.a = color(3);
        
        marker.lifetime = ros::Duration(lifetime);
        
        return marker;
    }

    visualization_msgs::Marker Visualizer::createCentroidMarker(
        const Eigen::Vector4f& centroid,
        int id,
        const std::string& frame_id,
        const Eigen::Vector4d& color,
        double scale,
        double lifetime)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "centroid";
        marker.id = id;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        
        marker.pose.position.x = centroid.x();
        marker.pose.position.y = centroid.y();
        marker.pose.position.z = centroid.z();
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = scale;
        marker.scale.y = scale;
        marker.scale.z = scale;
        
        marker.color.r = color(0);
        marker.color.g = color(1);
        marker.color.b = color(2);
        marker.color.a = color(3);
        
        marker.lifetime = ros::Duration(lifetime);
        
        return marker;
    }

    visualization_msgs::Marker Visualizer::createTextMarker(
        const Eigen::Vector3f& position,
        const std::string& text,
        int id,
        const std::string& frame_id,
        double scale,
        double lifetime)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "text";
        marker.id = id;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        marker.action = visualization_msgs::Marker::ADD;
        
        marker.pose.position.x = position.x();
        marker.pose.position.y = position.y();
        marker.pose.position.z = position.z() + 0.5;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.z = scale;
        
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;
        
        marker.text = text;
        marker.lifetime = ros::Duration(lifetime);
        
        return marker;
    }

    void Visualizer::broadcastTF(
        const Eigen::Vector3f& position,
        const Eigen::Quaternionf& orientation,
        const std::string& parent_frame,
        const std::string& child_frame,
        tf::TransformBroadcaster& broadcaster)
    {
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(position.x(), position.y(), position.z()));
        transform.setRotation(tf::Quaternion(orientation.x(), orientation.y(), 
                                            orientation.z(), orientation.w()));
        
        broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), 
                                                        parent_frame, child_frame));
    }

}  // namespace rm_radar_lidar_detector
