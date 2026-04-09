#include "lidar_detector.h"
#include <pluginlib/class_list_macros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rm_radar_msgs/DroneDetection.h>
#include <boost/make_shared.hpp>

PLUGINLIB_EXPORT_CLASS(rm_radar_lidar_detector::LidarDetector, nodelet::Nodelet)

namespace rm_radar_lidar_detector
{

    LidarDetector::~LidarDetector()
    {
        callback_queue_.disable();
        callback_queue_.clear();
        if (worker_thread_.joinable())
        {
            worker_thread_.join();
        }
    }

    void LidarDetector::onInit()
    {
        NODELET_INFO("LidarDetector initializing...");

        ros::NodeHandle& private_nh = getMTPrivateNodeHandle();
        private_nh.setCallbackQueue(&callback_queue_);

        initialize(private_nh);

        worker_thread_ = std::thread([this]() {
            ros::SingleThreadedSpinner spinner;
            spinner.spin(&callback_queue_);
        });
        
        NODELET_INFO("LidarDetector initialized successfully");
    }

    void LidarDetector::initialize(ros::NodeHandle& nh)
    {
        nh_ = nh;
        private_nh_ = nh;
        
        initParams();
        initROS();
        setupDynamicReconfigure();
    }

    void LidarDetector::initParams()
    {
        // Topic and frame
        private_nh_.param<std::string>("cloud_topic", cloud_topic_, "/livox/lidar");
        private_nh_.param<std::string>("frame_id", frame_id_, "livox_frame");
        
        FilterParams filter_params;
        private_nh_.param<float>("pass_x_min", filter_params.pass_x_min, 0.0f);
        private_nh_.param<float>("pass_x_max", filter_params.pass_x_max, 35.0f);
        private_nh_.param<float>("pass_y_min", filter_params.pass_y_min, -10.0f);
        private_nh_.param<float>("pass_y_max", filter_params.pass_y_max, 8.0f);
        private_nh_.param<float>("pass_z_min", filter_params.pass_z_min, -50.0f);
        private_nh_.param<float>("pass_z_max", filter_params.pass_z_max, 7.0f);
        private_nh_.param<float>("voxel_leaf_size", filter_params.voxel_leaf, 0.1f);
        private_nh_.param<float>("radius_search", filter_params.radius_search, 0.5f);
        private_nh_.param<int>("min_neighbors", filter_params.min_neighbors, 5);
        private_nh_.param<float>("ransac_distance_threshold", filter_params.ransac_distance_threshold, 0.5f);
        cloud_processor_.setParams(filter_params);
        
        DynamicDetectorParams dynamic_params;
        private_nh_.param<float>("distance_threshold", dynamic_params.distance_threshold, 2.0f);
        private_nh_.param<float>("aircraft_min_z", dynamic_params.min_z, -3.0f);
        dynamic_detector_.setParams(dynamic_params);
        
        ClusterFilterParams cluster_params;
        private_nh_.param<int>("min_n", cluster_params.min_n, cluster_params.min_n);
        private_nh_.param<int>("max_n", cluster_params.max_n, cluster_params.max_n);
        private_nh_.param<double>("min_v", cluster_params.min_v, cluster_params.min_v);
        private_nh_.param<double>("max_v", cluster_params.max_v, cluster_params.max_v);
        private_nh_.param<double>("min_r", cluster_params.min_r, cluster_params.min_r);
        private_nh_.param<double>("max_r", cluster_params.max_r, cluster_params.max_r);
        private_nh_.param<float>("min_h", cluster_params.min_h, cluster_params.min_h);
        private_nh_.param<float>("max_h", cluster_params.max_h, cluster_params.max_h);
        cluster_filter_.setParams(cluster_params);
        
        // Clustering parameters
        private_nh_.param<float>("eps", eps_, 0.5f);
        private_nh_.param<int>("minPts", minPts_, 5);
        private_nh_.param<float>("expansion_m", expansion_m_, 1.0f);
        
        int mode;
        private_nh_.param<int>("detection_mode", mode, 0);
        detection_mode_ = static_cast<DetectionMode>(mode);
        private_nh_.param<bool>("verbose_log", verbose_log_, false);
        private_nh_.param<bool>("publish_cluster_debug_markers", publish_cluster_debug_markers_, true);
        private_nh_.param<int>("accumulated_publish_divider", accumulated_publish_divider_, 2);
        if (accumulated_publish_divider_ < 1)
        {
            accumulated_publish_divider_ = 1;
        }
        cluster_tick_count_ = 0;
        
        private_nh_.param<int>("cloud_queue_size", cloud_queue_size_, 10);
        int frame_gap_int;
        private_nh_.param<int>("frame_gap", frame_gap_int, 4);
        frame_gap_ = static_cast<size_t>(frame_gap_int);
        cur_filtered_cloud_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        
        NODELET_INFO("Parameters initialized");
    }

    void LidarDetector::initROS()
    {
        // Subscribers
        cloud_sub_ = nh_.subscribe(cloud_topic_, 1, &LidarDetector::cloudCallback, this);
        
        // Publishers
        filtered_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
        accumulated_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("accumulated_cloud", 1);
        drone_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("drone_cloud", 1);
        marker_pub_ = nh_.advertise<visualization_msgs::Marker>("detection_marker", 10);
        cluster_debug_marker_array_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("cluster_debug_markers", 1);
        detection_pub_ = nh_.advertise<rm_radar_msgs::DroneDetection>("lidar_detection", 1);  // 供 rm_track 融合使用
        
        // Timer for clustering (10 Hz)
        cluster_timer_ = nh_.createTimer(ros::Duration(0.1), &LidarDetector::clusterTimerCallback, this);
        
        // Initialize visualizer
        visualizer_.init(nh_);
        
        NODELET_INFO("ROS interfaces initialized");
    }

    void LidarDetector::setupDynamicReconfigure()
    {
        dr_server_ = boost::make_shared<dynamic_reconfigure::Server<FilterParamsConfig>>(nh_);
        dr_server_->setCallback(boost::bind(&LidarDetector::dynamicReconfigureCallback, this, _1, _2));
    }

    void LidarDetector::dynamicReconfigureCallback(FilterParamsConfig& config, uint32_t level)
    {
        // Update filter parameters
        FilterParams filter_params = cloud_processor_.getParams();
        filter_params.voxel_leaf = config.voxel_leaf_size;
        filter_params.pass_x_min = config.pass_x_min;
        filter_params.pass_x_max = config.pass_x_max;
        filter_params.pass_y_min = config.pass_y_min;
        filter_params.pass_y_max = config.pass_y_max;
        filter_params.pass_z_min = config.pass_z_min;
        filter_params.pass_z_max = config.pass_z_max;
        filter_params.radius_search = config.radius_search;
        filter_params.min_neighbors = config.min_neighbors;
        filter_params.ransac_distance_threshold = config.ransac_distance_threshold;
        cloud_processor_.setParams(filter_params);
        
        // Update dynamic detector parameters
        DynamicDetectorParams dynamic_params = dynamic_detector_.getParams();
        dynamic_params.distance_threshold = config.distance_threshold;
        dynamic_params.min_z = config.aircraft_min_z;
        dynamic_detector_.setParams(dynamic_params);
        
        // Update cluster filter parameters
        ClusterFilterParams cluster_params = cluster_filter_.getParams();
        cluster_params.min_n = config.min_n;
        cluster_params.max_n = config.max_n;
        cluster_params.min_v = config.min_v;
        cluster_params.max_v = config.max_v;
        cluster_params.min_r = config.min_r;
        cluster_params.max_r = config.max_r;
        cluster_params.min_h = config.min_h;
        cluster_params.max_h = config.max_h;
        cluster_filter_.setParams(cluster_params);
        
        // Update clustering parameters
        eps_ = config.eps;
        minPts_ = config.minPts;
        expansion_m_ = config.expansion_m;
        
        // Update detection mode
        DetectionMode new_mode = static_cast<DetectionMode>(config.detection_mode);
        if (new_mode != detection_mode_) {
            detection_mode_ = new_mode;
            if (verbose_log_)
            {
                NODELET_INFO("Detection mode changed to: %s",
                            detection_mode_ == DetectionMode::DYNAMIC ? "DYNAMIC" : "DIRECT");
            }
        }
        
        if (verbose_log_)
        {
            NODELET_INFO("Dynamic reconfigure: voxel=%.3f, eps=%.3f, minPts=%d, mode=%s",
                        filter_params.voxel_leaf, eps_, minPts_,
                        detection_mode_ == DetectionMode::DYNAMIC ? "DYNAMIC" : "DIRECT");
        }
    }

    void LidarDetector::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(*cloud_msg, *cloud);
        
        if (cloud->empty()) {
            NODELET_WARN_THROTTLE(5, "Received empty point cloud");
            return;
        }
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = cloud_processor_.preprocess(cloud);
        
        if (filtered->empty()) {
            NODELET_WARN_THROTTLE(5, "Filtered cloud is empty");
            return;
        }
        
        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            cloud_queue_.push_back(filtered);
            while (cloud_queue_.size() > static_cast<size_t>(cloud_queue_size_)) {
                cloud_queue_.pop_front();
            }
            cur_filtered_cloud_ = filtered;
            frame_id_ = cloud_msg->header.frame_id;
            latest_cloud_stamp_ = cloud_msg->header.stamp;
        }
        
        visualizer_.publishCloud(filtered, frame_id_, cloud_msg->header.stamp, filtered_cloud_pub_);
    }

    void LidarDetector::clusterTimerCallback(const ros::TimerEvent& event)
    {
        std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> queue_snapshot;
        pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_snapshot;
        std::string frame_id_snapshot;
        ros::Time stamp_snapshot;

        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            queue_snapshot = cloud_queue_;
            frame_id_snapshot = frame_id_;
            stamp_snapshot = latest_cloud_stamp_;
        }
        ++cluster_tick_count_;

        const bool need_accumulated_for_detection = (detection_mode_ == DetectionMode::DIRECT);
        const bool need_accumulated_for_publish = (accumulated_cloud_pub_.getNumSubscribers() > 0);
        const bool need_accumulated = need_accumulated_for_detection || need_accumulated_for_publish;
        const bool publish_accumulated =
            need_accumulated_for_publish && ((cluster_tick_count_ % static_cast<uint64_t>(accumulated_publish_divider_)) == 0);
        if (need_accumulated && !queue_snapshot.empty())
        {
            accumulated_snapshot = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            for (const auto& c : queue_snapshot)
            {
                if (c && !c->empty())
                {
                    *accumulated_snapshot += *c;
                }
            }
            if (publish_accumulated && accumulated_snapshot && !accumulated_snapshot->empty())
            {
                const ros::Time accum_stamp = stamp_snapshot.isZero() ? ros::Time::now() : stamp_snapshot;
                visualizer_.publishCloud(accumulated_snapshot, frame_id_snapshot, accum_stamp, accumulated_cloud_pub_);
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr detected_cluster;
        BBox3D detected_bbox;
        Eigen::Vector4f detected_centroid = Eigen::Vector4f::Zero();
        std::vector<ClusterDebugInfo> cluster_debug_infos;

        if (detection_mode_ == DetectionMode::DYNAMIC)
        {
            if (queue_snapshot.size() < frame_gap_ + 1)
            {
                if (verbose_log_)
                {
                    NODELET_DEBUG_THROTTLE(1, "DYNAMIC: Waiting for more frames (%zu/%zu)",
                                           queue_snapshot.size(), frame_gap_ + 1);
                }
                return;
            }

            const auto current_cloud = queue_snapshot.back();
            auto it = queue_snapshot.begin();
            std::advance(it, queue_snapshot.size() - frame_gap_ - 1);
            const auto reference_cloud = *it;

            if (!current_cloud || current_cloud->empty() ||
                !reference_cloud || reference_cloud->empty())
            {
                return;
            }

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            kdtree.setInputCloud(reference_cloud);

            DynamicDetectorParams params = dynamic_detector_.getParams();
            dynamic_points_buffer_.clear();
            dynamic_points_buffer_.reserve(current_cloud->size());
            nn_indices_buffer_.clear();
            nn_distances_buffer_.clear();
            for (const auto& pt : current_cloud->points)
            {
                if (pt.z < params.min_z)
                {
                    continue;
                }

                nn_indices_buffer_.clear();
                nn_distances_buffer_.clear();
                if (kdtree.radiusSearch(pt, params.distance_threshold, nn_indices_buffer_, nn_distances_buffer_) == 0)
                {
                    dynamic_points_buffer_.emplace_back(pt.x, pt.y, pt.z);
                }
            }

            if (verbose_log_)
            {
                NODELET_INFO_THROTTLE(1, "DYNAMIC: current=%zu, ref=%zu, dynamic=%zu, threshold=%.2f",
                                      current_cloud->size(), reference_cloud->size(),
                                      dynamic_points_buffer_.size(), params.distance_threshold);
            }

            if (dynamic_points_buffer_.size() < static_cast<size_t>(minPts_))
            {
                return;
            }

            DBSCAN dbscan(dynamic_points_buffer_, eps_, minPts_);
            int num_clusters = dbscan.run();
            if (num_clusters == 0)
            {
                return;
            }

            if (verbose_log_)
            {
                NODELET_INFO("DYNAMIC: Found %d clusters from %zu dynamic points", num_clusters, dynamic_points_buffer_.size());
            }
            detected_cluster = cluster_filter_.findBest(dbscan.getClusters(), detected_bbox, detected_centroid, &cluster_debug_infos);
        }
        else
        {
            if (!accumulated_snapshot || accumulated_snapshot->empty())
            {
                return;
            }

            DynamicDetectorParams params = dynamic_detector_.getParams();
            airspace_points_buffer_.clear();
            airspace_points_buffer_.reserve(accumulated_snapshot->size());

            for (const auto& pt : accumulated_snapshot->points)
            {
                if (pt.z >= params.min_z)
                {
                    airspace_points_buffer_.emplace_back(pt.x, pt.y, pt.z);
                }
            }

            if (verbose_log_)
            {
                NODELET_INFO_THROTTLE(1, "DIRECT: accumulated=%zu, airspace=%zu (min_z=%.1f)",
                                      accumulated_snapshot->size(), airspace_points_buffer_.size(), params.min_z);
            }

            if (airspace_points_buffer_.size() < static_cast<size_t>(minPts_))
            {
                return;
            }

            DBSCAN dbscan(airspace_points_buffer_, eps_, minPts_);
            int num_clusters = dbscan.run();
            if (num_clusters == 0)
            {
                return;
            }

            if (verbose_log_)
            {
                NODELET_INFO("DIRECT: Found %d clusters from %zu airspace points", num_clusters, airspace_points_buffer_.size());
            }
            detected_cluster = cluster_filter_.findBest(dbscan.getClusters(), detected_bbox, detected_centroid, &cluster_debug_infos);
        }

        const ros::Time debug_stamp = stamp_snapshot.isZero() ? ros::Time::now() : stamp_snapshot;
        if (publish_cluster_debug_markers_)
        {
            publishClusterDebugMarkers(cluster_debug_infos, frame_id_snapshot, debug_stamp);
        }

        if (detected_cluster && !detected_cluster->empty())
        {
            const ros::Time detection_stamp = stamp_snapshot.isZero() ? ros::Time::now() : stamp_snapshot;
            publishResults(detected_cluster, detected_bbox, detected_centroid, frame_id_snapshot, detection_stamp);
        }
    }

    void LidarDetector::publishResults(const pcl::PointCloud<pcl::PointXYZ>::Ptr& detected_cluster,
                                       const BBox3D& detected_bbox,
                                       const Eigen::Vector4f& detected_centroid,
                                       const std::string& frame_id,
                                       const ros::Time& stamp)
    {
        if (detected_cluster && !detected_cluster->empty()) {
            visualizer_.publishCloud(detected_cluster, frame_id, stamp, drone_cloud_pub_);
            
            auto marker = Visualizer::createAABBMarker(
                detected_bbox, 0, "detected_cluster", frame_id,
                Eigen::Vector4d(1.0, 0.0, 0.0, 0.5));
            marker_pub_.publish(marker);
            
            auto centroid_marker = Visualizer::createCentroidMarker(
                detected_centroid, 100, frame_id,
                Eigen::Vector4d(1.0, 1.0, 0.0, 1.0), 0.2);
            marker_pub_.publish(centroid_marker);
            
            rm_radar_msgs::DroneDetection detection_msg;
            detection_msg.header.stamp = stamp;
            detection_msg.header.frame_id = frame_id;
            detection_msg.is_cam_msg = false;
            detection_msg.is_lidar_msg = true;
            
            detection_msg.pose.position.x = detected_centroid.x();
            detection_msg.pose.position.y = detected_centroid.y();
            detection_msg.pose.position.z = detected_centroid.z();
            detection_msg.pose.orientation.w = 1.0;
            detection_msg.pose.orientation.x = 0.0;
            detection_msg.pose.orientation.y = 0.0;
            detection_msg.pose.orientation.z = 0.0;
            
            detection_msg.centroid.x = detected_centroid.x();
            detection_msg.centroid.y = detected_centroid.y();
            detection_msg.centroid.z = detected_centroid.z();
            
            detection_pub_.publish(detection_msg);
            
            NODELET_DEBUG("Published detection: centroid=(%.2f, %.2f, %.2f)",
                        detected_centroid.x(), detected_centroid.y(), detected_centroid.z());
        }
    }

    void LidarDetector::publishClusterDebugMarkers(const std::vector<ClusterDebugInfo>& debug_infos,
                                                   const std::string& frame_id,
                                                   const ros::Time& stamp)
    {
        if (cluster_debug_marker_array_pub_.getNumSubscribers() == 0)
        {
            return;
        }
        visualization_msgs::MarkerArray marker_array =
            Visualizer::createClusterDebugMarkerArray(debug_infos, frame_id, stamp, 0.25, 0.25);
        cluster_debug_marker_array_pub_.publish(marker_array);
    }

}  // namespace rm_radar_lidar_detector
