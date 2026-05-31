#include "visualizer.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

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

    visualization_msgs::MarkerArray Visualizer::createClusterDebugMarkerArray(
        const std::vector<ClusterDebugInfo>& debug_infos,
        const std::string& frame_id,
        const ros::Time& stamp,
        double box_lifetime,
        double text_lifetime)
    {
        visualization_msgs::MarkerArray marker_array;

        visualization_msgs::Marker clear_marker;
        clear_marker.header.frame_id = frame_id;
        clear_marker.header.stamp = stamp;
        clear_marker.ns = "cluster_debug_text";
        clear_marker.id = 0;
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        int marker_id = 1;
        for (const auto& info : debug_infos)
        {
            const Eigen::Vector4d bbox_color = info.is_best
                ? Eigen::Vector4d(1.0, 1.0, 0.0, 0.45)
                : (info.valid
                    ? Eigen::Vector4d(0.0, 1.0, 0.0, 0.35)
                    : Eigen::Vector4d(1.0, 0.0, 0.0, 0.35));
            auto bbox_marker = createAABBMarker(
                info.bbox,
                marker_id++,
                "cluster_debug_bbox",
                frame_id,
                bbox_color,
                box_lifetime);
            bbox_marker.header.stamp = stamp;
            marker_array.markers.push_back(bbox_marker);

            Eigen::Vector3f pos(info.centroid.x(), info.centroid.y(), info.centroid.z());
            std::ostringstream oss;
            oss << (info.is_best ? "[BEST] " : (info.valid ? "[OK] " : "[REJ] "))
                << "id=" << info.cluster_index
                << " n=" << info.point_count
                << " v=" << std::fixed << std::setprecision(2) << info.volume
                << " r=" << std::fixed << std::setprecision(2) << info.ratio
                << " p=" << std::fixed << std::setprecision(2) << info.pca_ratio
                << " h=" << std::fixed << std::setprecision(2) << info.centroid.z()
                << " s=" << std::fixed << std::setprecision(2) << info.score;

            auto marker = createTextMarker(pos, oss.str(), marker_id++, frame_id, 0.25, text_lifetime);
            marker.header.stamp = stamp;
            marker.ns = "cluster_debug_text";
            marker.color.r = info.is_best ? 1.0f : (info.valid ? 0.0f : 1.0f);
            marker.color.g = info.is_best ? 1.0f : (info.valid ? 1.0f : 0.0f);
            marker.color.b = 0.0f;
            marker.color.a = 1.0f;
            marker_array.markers.push_back(marker);
        }

        return marker_array;
    }

    visualization_msgs::Marker Visualizer::createDeleteMarker(
        const std::string& frame_id,
        const std::string& ns,
        int id,
        const ros::Time& stamp)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = stamp;
        marker.ns = ns;
        marker.id = id;
        marker.action = visualization_msgs::Marker::DELETE;
        return marker;
    }

    visualization_msgs::Marker Visualizer::createTrackTargetSphereMarker(
        const SingleTargetTracker::State& state,
        bool has_track,
        const std::string& frame_id,
        int id,
        const ros::Time& stamp,
        double lifetime,
        double sphere_scale)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = stamp;
        marker.ns = "track_target";
        marker.id = id;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.lifetime = ros::Duration(lifetime);

        if (!has_track)
        {
            marker.action = visualization_msgs::Marker::DELETE;
            return marker;
        }

        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = state.position.x();
        marker.pose.position.y = state.position.y();
        marker.pose.position.z = state.position.z();
        marker.pose.orientation.w = 1.0;
        marker.scale.x = sphere_scale;
        marker.scale.y = sphere_scale;
        marker.scale.z = sphere_scale;
        marker.color.r = 0.0f;
        marker.color.g = 0.95f;
        marker.color.b = 0.1f;
        marker.color.a = 0.90f;
        return marker;
    }

    visualization_msgs::Marker Visualizer::createTrackVelocityArrowMarker(
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
        double max_length)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = stamp;
        marker.ns = "track_target";
        marker.id = id;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.lifetime = ros::Duration(lifetime);

        if (!has_track)
        {
            marker.action = visualization_msgs::Marker::DELETE;
            return marker;
        }

        const float speed = state.velocity.norm();
        if (speed <= static_cast<float>(min_speed))
        {
            marker.action = visualization_msgs::Marker::DELETE;
            return marker;
        }

        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = shaft_diameter;
        marker.scale.y = head_diameter;
        marker.scale.z = head_length;
        marker.color.r = 1.0f;
        marker.color.g = 0.85f;
        marker.color.b = 0.1f;
        marker.color.a = 0.95f;

        geometry_msgs::Point p0;
        p0.x = state.position.x();
        p0.y = state.position.y();
        p0.z = state.position.z();
        geometry_msgs::Point p1 = p0;
        const float arrow_len = std::min(static_cast<float>(max_length), speed * static_cast<float>(length_scale));
        p1.x += state.velocity.x() / speed * arrow_len;
        p1.y += state.velocity.y() / speed * arrow_len;
        p1.z += state.velocity.z() / speed * arrow_len;
        marker.points.push_back(p0);
        marker.points.push_back(p1);
        return marker;
    }

    visualization_msgs::Marker Visualizer::createTrackTrajectoryMarker(
        const SingleTargetTracker::State& state,
        bool has_track,
        bool publish_trajectory,
        const std::string& frame_id,
        int id,
        const ros::Time& stamp,
        double line_width,
        float min_step,
        size_t max_points,
        std::deque<geometry_msgs::Point>& trajectory_points)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = stamp;
        marker.ns = "track_target";
        marker.id = id;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.lifetime = ros::Duration(0.0);

        if (!publish_trajectory)
        {
            marker.action = visualization_msgs::Marker::DELETE;
            trajectory_points.clear();
            return marker;
        }

        if (!has_track)
        {
            marker.action = visualization_msgs::Marker::DELETE;
            trajectory_points.clear();
            return marker;
        }

        geometry_msgs::Point curr;
        curr.x = state.position.x();
        curr.y = state.position.y();
        curr.z = state.position.z();

        if (trajectory_points.empty())
        {
            trajectory_points.push_back(curr);
        }
        else
        {
            const auto& last = trajectory_points.back();
            const float dx = static_cast<float>(curr.x - last.x);
            const float dy = static_cast<float>(curr.y - last.y);
            const float dz = static_cast<float>(curr.z - last.z);
            const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (dist >= min_step)
            {
                trajectory_points.push_back(curr);
            }
        }

        while (trajectory_points.size() > max_points)
        {
            trajectory_points.pop_front();
        }

        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = line_width;
        marker.color.r = 0.10f;
        marker.color.g = 0.95f;
        marker.color.b = 1.00f;
        marker.color.a = 0.90f;
        marker.points.assign(trajectory_points.begin(), trajectory_points.end());
        return marker;
    }

    visualization_msgs::Marker Visualizer::createTrackSpeedTextMarker(
        const SingleTargetTracker::State& state,
        bool has_track,
        const std::string& frame_id,
        int id,
        const ros::Time& stamp,
        double lifetime,
        float z_offset,
        float text_scale)
    {
        visualization_msgs::Marker marker;
        marker.header.stamp = stamp;
        marker.header.frame_id = frame_id;
        marker.ns = "track_text";
        marker.id = id;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        marker.lifetime = ros::Duration(lifetime);

        if (!has_track)
        {
            marker.action = visualization_msgs::Marker::DELETE;
            return marker;
        }

        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = state.position.x();
        marker.pose.position.y = state.position.y();
        marker.pose.position.z = state.position.z() + z_offset;
        marker.pose.orientation.w = 1.0;
        marker.scale.z = text_scale;
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 1.0f;
        marker.color.a = 0.95f;

        std::ostringstream ss;
        ss << "v=" << std::fixed << std::setprecision(2) << static_cast<double>(state.velocity.norm()) << "m/s";
        marker.text = ss.str();
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
