#include "lidar_detector.h"
#include <pluginlib/class_list_macros.h>
#include "dbscan.h"
#include <rm_radar_msgs/DroneDetection.h>
#include <rm_radar_msgs/DroneTrackData.h>
#include <boost/make_shared.hpp>
#include <pcl/kdtree/kdtree_flann.h>

#include <algorithm>
#include <cmath>

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
            private_nh_.param<bool>("enable_voxel_downsample", filter_params.enable_voxel_downsample, true);
            private_nh_.param<float>("voxel_leaf_size", filter_params.voxel_leaf, 0.1f);
            private_nh_.param<float>("radius_search", filter_params.radius_search, 0.5f);
            private_nh_.param<int>("min_neighbors", filter_params.min_neighbors, 5);
            private_nh_.param<bool>("remove_planes", filter_params.remove_planes, filter_params.remove_planes);
            private_nh_.param<float>("plane_distance_threshold", filter_params.plane_distance_threshold, filter_params.plane_distance_threshold);
            private_nh_.param<int>("plane_max_iterations", filter_params.plane_max_iterations, filter_params.plane_max_iterations);
            private_nh_.param<int>("plane_max_planes", filter_params.plane_max_planes, filter_params.plane_max_planes);
            private_nh_.param<int>("plane_min_points", filter_params.plane_min_points, filter_params.plane_min_points);
            private_nh_.param<float>("plane_min_inlier_ratio", filter_params.plane_min_inlier_ratio, filter_params.plane_min_inlier_ratio);
            private_nh_.param<int>("plane_fail_streak_threshold", filter_params.plane_fail_streak_threshold, filter_params.plane_fail_streak_threshold);
            private_nh_.param<int>("plane_fail_cooldown_frames", filter_params.plane_fail_cooldown_frames, filter_params.plane_fail_cooldown_frames);
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
            private_nh_.param<double>("min_pca_r", cluster_params.min_pca_r, cluster_params.min_pca_r);
            private_nh_.param<double>("max_pca_r", cluster_params.max_pca_r, cluster_params.max_pca_r);
            private_nh_.param<float>("min_h", cluster_params.min_h, cluster_params.min_h);
            private_nh_.param<float>("max_h", cluster_params.max_h, cluster_params.max_h);
            cluster_filter_.setParams(cluster_params);
            
            // Clustering parameters
            private_nh_.param<float>("eps", eps_, 0.5f);
            private_nh_.param<int>("minPts", minPts_, 5);
            
            int mode;
            private_nh_.param<int>("detection_mode", mode, 0);
            detection_mode_ = static_cast<DetectionMode>(mode);
            private_nh_.param<bool>("verbose_log", verbose_log_, false);
            private_nh_.param<bool>("publish_cluster_debug_markers", publish_cluster_debug_markers_, true);
            private_nh_.param<bool>("publish_plane_debug_clouds", publish_plane_debug_clouds_, true);
            private_nh_.param<bool>("track_enable", track_enable_, true);
            private_nh_.param<bool>("publish_track_text", publish_track_text_, true);
            private_nh_.param<bool>("publish_track_target_marker", publish_track_target_marker_, true);
            private_nh_.param<bool>("publish_track_trajectory", publish_track_trajectory_, true);
            private_nh_.param<bool>("publish_raw_drone_cloud", publish_raw_drone_cloud_, true);
            private_nh_.param<bool>("extract_target_from_input_cloud", extract_target_from_input_cloud_, false);
            private_nh_.param<bool>("project_input_cloud_directly", project_input_cloud_directly_, false);
            private_nh_.param<double>("selected_target_point_marker_scale", selected_target_point_marker_scale_, 0.01);
            private_nh_.param<double>("target_point_smoothing_alpha", target_point_smoothing_alpha_, 1.0);
            private_nh_.param<double>("target_point_max_jump_distance", target_point_max_jump_distance_, 0.8);
            target_point_smoothing_alpha_ = std::clamp(target_point_smoothing_alpha_, 0.0, 1.0);
            target_point_max_jump_distance_ = std::max(0.0, target_point_max_jump_distance_);
            private_nh_.param<int>("accumulated_publish_divider", accumulated_publish_divider_, 2);
            private_nh_.param<int>("drone_aabb_accumulation_max_points", drone_aabb_accumulation_max_points_, 3000);
            private_nh_.param<double>("drone_aabb_accumulation_reset_distance", drone_aabb_accumulation_reset_distance_, 0.25);
            drone_aabb_accumulation_reset_distance_ = std::max(0.0, drone_aabb_accumulation_reset_distance_);
            cloud_processor_.setAabbLocalAccumulationResetDistance(static_cast<float>(drone_aabb_accumulation_reset_distance_));
            TargetFrontViewController::Params front_view_params = target_front_view_controller_.getParams();
            private_nh_.param<bool>("show_target_front_view", front_view_params.enabled, front_view_params.enabled);
            private_nh_.param<std::string>("target_front_view_window", front_view_params.front_view_window, front_view_params.front_view_window);
            private_nh_.param<std::string>("target_template_window", front_view_params.template_window, front_view_params.template_window);
            private_nh_.param<int>("target_front_view_accumulation_frames", front_view_params.accumulation_frames, front_view_params.accumulation_frames);
            private_nh_.param<float>("target_front_view_resolution", front_view_params.projector.grid_resolution, front_view_params.projector.grid_resolution);
            private_nh_.param<bool>("target_front_view_adaptive_resolution",
                                    front_view_params.projector.adaptive_resolution,
                                    front_view_params.projector.adaptive_resolution);
            private_nh_.param<float>("target_front_view_resolution_scale",
                                    front_view_params.projector.adaptive_resolution_scale,
                                    front_view_params.projector.adaptive_resolution_scale);
            private_nh_.param<float>("target_front_view_max_resolution",
                                    front_view_params.projector.max_grid_resolution,
                                    front_view_params.projector.max_grid_resolution);
            private_nh_.param<int>("target_front_view_max_image_size", front_view_params.projector.max_image_size, front_view_params.projector.max_image_size);
            private_nh_.param<int>("target_front_view_min_points", front_view_params.projector.min_points, front_view_params.projector.min_points);
            private_nh_.param<double>("target_template_update_min_score",
                                     front_view_params.patch_update_min_score,
                                     front_view_params.patch_update_min_score);
            private_nh_.param<double>("target_template_learning_rate",
                                     front_view_params.patch_learning_rate,
                                     front_view_params.patch_learning_rate);
            private_nh_.param<double>("target_template_train_threshold",
                                     front_view_params.patch_train_threshold,
                                     front_view_params.patch_train_threshold);
            private_nh_.param<int>("target_template_train_min_points",
                                  front_view_params.patch_train_min_points,
                                  front_view_params.patch_train_min_points);
            target_front_view_controller_.setSelectedPointCallback(
                [this](const pcl::PointXYZ& point, const std::string& frame_id, const ros::Time& stamp) {
                    publishSelectedTargetPoint(point, frame_id, stamp);
                });
            target_front_view_controller_.setParams(front_view_params);

            TargetCloudExtractor::Params target_extractor_params = target_cloud_extractor_.getParams();
            private_nh_.param<int>("target_cloud_min_support_points",
                                target_extractor_params.min_support_points,
                                target_extractor_params.min_support_points);
            private_nh_.param<float>("target_cloud_height_window",
                                    target_extractor_params.height_window,
                                    target_extractor_params.height_window);
            target_cloud_extractor_.setParams(target_extractor_params);
            if (accumulated_publish_divider_ < 1)
            {
                accumulated_publish_divider_ = 1;
            }
            drone_aabb_accumulation_max_points_ = std::max(100, drone_aabb_accumulation_max_points_);
            cluster_tick_count_ = 0;
            
            private_nh_.param<int>("cloud_queue_size", cloud_queue_size_, 10);
            int frame_gap_int;
            private_nh_.param<int>("frame_gap", frame_gap_int, 4);
            frame_gap_ = static_cast<size_t>(frame_gap_int);
            cur_filtered_cloud_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

            SingleTargetTracker::Params tracker_params;
            private_nh_.param<float>("track_gate_distance_m", tracker_params.gate_distance_m, tracker_params.gate_distance_m);
            private_nh_.param<bool>("track_local_cluster_enable", track_local_cluster_enable_, true);
            private_nh_.param<float>("track_max_lost_time_s", tracker_params.max_lost_time_s, tracker_params.max_lost_time_s);
            private_nh_.param<float>("track_pos_alpha", tracker_params.pos_alpha, tracker_params.pos_alpha);
            private_nh_.param<float>("track_vel_alpha", tracker_params.vel_alpha, tracker_params.vel_alpha);
            private_nh_.param<float>("track_max_speed_mps", tracker_params.max_speed_mps, tracker_params.max_speed_mps);
            private_nh_.param<float>("track_velocity_min_dt_s", tracker_params.velocity_min_dt_s, tracker_params.velocity_min_dt_s);
            private_nh_.param<float>("track_predict_max_dt_s", tracker_params.predict_max_dt_s, tracker_params.predict_max_dt_s);
            private_nh_.param<float>("track_miss_velocity_decay", tracker_params.miss_velocity_decay, tracker_params.miss_velocity_decay);
            tracker_.setParams(tracker_params);

            SingleTargetTracker::AssociationParams assoc_params;
            private_nh_.param<double>("confidence_on", assoc_params.conf_on, assoc_params.conf_on);
            private_nh_.param<double>("confidence_keep", assoc_params.conf_keep, assoc_params.conf_keep);
            private_nh_.param<int>("lock_min_streak", assoc_params.lock_min_streak, assoc_params.lock_min_streak);
            private_nh_.param<double>("gate_pos_base", assoc_params.gate_pos_base, assoc_params.gate_pos_base);
            private_nh_.param<double>("gate_pos_range_k", assoc_params.gate_pos_range_k, assoc_params.gate_pos_range_k);
            tracker_.setAssociationParams(assoc_params);
            
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
            target_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("target_cloud", 1);
            target_projected_surface_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("target_projected_surface_cloud", 1);
            drone_aabb_accumulated_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("drone_aabb_accumulated_cloud", 1);
            target_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("target_marker", 1);
            selected_target_point_pub_ = nh_.advertise<visualization_msgs::Marker>("selected_target_point", 1, true);
            marker_pub_ = nh_.advertise<visualization_msgs::Marker>("detection_marker", 10);
            cluster_debug_marker_array_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("cluster_debug_markers", 1);
            detection_pub_ = nh_.advertise<rm_radar_msgs::DroneDetection>("lidar_detection", 1);  // 供 rm_track 融合使用
            track_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("track_marker", 10);
            track_pub_ = nh_.advertise<rm_radar_msgs::DroneTrackData>("lidar_track", 1);
            
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
            filter_params.enable_voxel_downsample = config.enable_voxel_downsample;
            filter_params.voxel_leaf = config.voxel_leaf_size;
            filter_params.pass_x_min = config.pass_x_min;
            filter_params.pass_x_max = config.pass_x_max;
            filter_params.pass_y_min = config.pass_y_min;
            filter_params.pass_y_max = config.pass_y_max;
            filter_params.pass_z_min = config.pass_z_min;
            filter_params.pass_z_max = config.pass_z_max;
            filter_params.radius_search = config.radius_search;
            filter_params.min_neighbors = config.min_neighbors;
            filter_params.remove_planes = config.remove_planes;
            filter_params.plane_distance_threshold = static_cast<float>(config.plane_distance_threshold);
            filter_params.plane_max_iterations = config.plane_max_iterations;
            filter_params.plane_max_planes = config.plane_max_planes;
            filter_params.plane_min_points = config.plane_min_points;
            filter_params.plane_min_inlier_ratio = static_cast<float>(config.plane_min_inlier_ratio);
            filter_params.plane_fail_streak_threshold = config.plane_fail_streak_threshold;
            filter_params.plane_fail_cooldown_frames = config.plane_fail_cooldown_frames;
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
            cluster_params.min_pca_r = config.min_pca_r;
            cluster_params.max_pca_r = config.max_pca_r;
            cluster_params.min_h = config.min_h;
            cluster_params.max_h = config.max_h;
            cluster_filter_.setParams(cluster_params);
            
            // Update clustering parameters
            eps_ = config.eps;
            minPts_ = config.minPts;
            publish_plane_debug_clouds_ = config.publish_plane_debug_clouds;

            {
                std::lock_guard<std::mutex> lock(cloud_mutex_);
                cloud_queue_size_ = std::max(1, config.cloud_queue_size);
                while (cloud_queue_.size() > static_cast<std::size_t>(cloud_queue_size_))
                {
                    cloud_queue_.pop_front();
                }
                while (raw_cloud_queue_.size() > static_cast<std::size_t>(cloud_queue_size_))
                {
                    raw_cloud_queue_.pop_front();
                }
            }

            drone_aabb_accumulation_max_points_ = std::max(100, config.drone_aabb_accumulation_max_points);
            drone_aabb_accumulation_reset_distance_ = std::max(0.0, config.drone_aabb_accumulation_reset_distance);
            cloud_processor_.setAabbLocalAccumulationResetDistance(static_cast<float>(drone_aabb_accumulation_reset_distance_));
            cloud_processor_.trimAabbLocalAccumulation(drone_aabb_accumulation_max_points_);

            track_enable_ = config.track_enable;
            track_local_cluster_enable_ = config.track_local_cluster_enable;
            publish_track_text_ = config.publish_track_text;
            publish_track_target_marker_ = config.publish_track_target_marker;
            publish_track_trajectory_ = config.publish_track_trajectory;
            publish_raw_drone_cloud_ = config.publish_raw_drone_cloud;
            selected_target_point_marker_scale_ = std::max(0.001, config.selected_target_point_marker_scale);
            target_point_smoothing_alpha_ = std::clamp(config.target_point_smoothing_alpha, 0.0, 1.0);
            target_point_max_jump_distance_ = std::max(0.0, config.target_point_max_jump_distance);
            TargetFrontViewController::Params front_view_params = target_front_view_controller_.getParams();
            const int new_target_front_view_accumulation_frames = std::max(1, config.target_front_view_accumulation_frames);
            front_view_params.patch_update_min_score = std::clamp(config.target_template_update_min_score,
                                                                  front_view_params.patch_min_score,
                                                                  1.0);
            front_view_params.patch_learning_rate = std::clamp(config.target_template_learning_rate, 0.0, 1.0);
            front_view_params.patch_train_threshold = std::clamp(config.target_template_train_threshold, 0.0, 1.0);
            front_view_params.patch_train_min_points = std::max(1, config.target_template_train_min_points);
            if (new_target_front_view_accumulation_frames != front_view_params.accumulation_frames)
            {
                front_view_params.accumulation_frames = new_target_front_view_accumulation_frames;
                target_front_view_controller_.resetAccumulation();
            }
            target_front_view_controller_.setParams(front_view_params);

            TargetCloudExtractor::Params target_extractor_params = target_cloud_extractor_.getParams();
            target_extractor_params.min_support_points = std::max(2, config.target_cloud_min_support_points);
            target_extractor_params.height_window = static_cast<float>(std::max(0.02, config.target_cloud_height_window));
            target_cloud_extractor_.setParams(target_extractor_params);

            SingleTargetTracker::Params tracker_params = tracker_.getParams();
            tracker_params.gate_distance_m = static_cast<float>(config.track_gate_distance_m);
            tracker_params.max_lost_time_s = static_cast<float>(config.track_max_lost_time_s);
            tracker_params.pos_alpha = static_cast<float>(config.track_pos_alpha);
            tracker_params.vel_alpha = static_cast<float>(config.track_vel_alpha);
            tracker_params.max_speed_mps = static_cast<float>(config.track_max_speed_mps);
            tracker_params.velocity_min_dt_s = static_cast<float>(config.track_velocity_min_dt_s);
            tracker_params.predict_max_dt_s = static_cast<float>(config.track_predict_max_dt_s);
            tracker_params.miss_velocity_decay = static_cast<float>(config.track_miss_velocity_decay);
            tracker_.setParams(tracker_params);

            SingleTargetTracker::AssociationParams assoc_params = tracker_.getAssociationParams();
            assoc_params.conf_on = config.confidence_on;
            assoc_params.conf_keep = config.confidence_keep;
            assoc_params.lock_min_streak = config.lock_min_streak;
            assoc_params.gate_pos_base = config.gate_pos_base;
            assoc_params.gate_pos_range_k = config.gate_pos_range_k;
            tracker_.setAssociationParams(assoc_params);
            
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
        PointCloudPtr cloud = boost::make_shared<PointCloud>();
        pcl::fromROSMsg(*cloud_msg, *cloud);

        if (cloud->empty())
        {
            NODELET_WARN_THROTTLE(5, "Received empty point cloud");
            return;
        }

        processInputCloud(cloud, cloud_msg->header.frame_id, cloud_msg->header.stamp);
    }


    void LidarDetector::clusterTimerCallback(const ros::TimerEvent& /*event*/)
    {
        const DetectionInput input = processQueuedClouds();
        DroneCandidate drone = findDrone(input);
        const ros::Time stamp = input.stamp.isZero() ? ros::Time::now() : input.stamp;

        handleDroneCandidate(drone, input.frame_id, stamp);
    }


    void LidarDetector::processInputCloud(const PointCloudPtr& cloud,
                                        const std::string& frame_id,
                                        const ros::Time& stamp)
    {
        if (project_input_cloud_directly_)
        {
            visualizer_.publishCloud(cloud, frame_id, stamp, target_cloud_pub_);
            const auto front_view_output = target_front_view_controller_.show(cloud, frame_id, stamp);
            if (front_view_output.projected_surface_cloud && !front_view_output.projected_surface_cloud->empty())
            {
                visualizer_.publishCloud(front_view_output.projected_surface_cloud,
                                        frame_id,
                                        stamp,
                                        target_projected_surface_cloud_pub_);
            }
            return;
        }

        if (extract_target_from_input_cloud_)
        {
            findTarget(cloud, frame_id, stamp);
            return;
        }

        PointCloudPtr filtered = cloud_processor_.preprocess(cloud);
        if (filtered->empty())
        {
            NODELET_WARN_THROTTLE(5, "Filtered cloud is empty");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            cloud_queue_.push_back(filtered);
            raw_cloud_queue_.push_back(cloud);
            while (cloud_queue_.size() > static_cast<size_t>(cloud_queue_size_))
            {
                cloud_queue_.pop_front();
            }
            while (raw_cloud_queue_.size() > static_cast<size_t>(cloud_queue_size_))
            {
                raw_cloud_queue_.pop_front();
            }
            cur_filtered_cloud_ = filtered;
            frame_id_ = frame_id;
            latest_cloud_stamp_ = stamp;
        }

        visualizer_.publishCloud(filtered, frame_id_, stamp, filtered_cloud_pub_);
    }


    LidarDetector::DetectionInput LidarDetector::processQueuedClouds()
    {
        DetectionInput input;
        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            input.cloud_queue = cloud_queue_;
            input.raw_cloud_queue = raw_cloud_queue_;
            input.frame_id = frame_id_;
            input.stamp = latest_cloud_stamp_;
        }
        ++cluster_tick_count_;

        const bool need_accumulated_for_detection = (detection_mode_ == DetectionMode::DIRECT);
        const bool need_accumulated_for_publish = (accumulated_cloud_pub_.getNumSubscribers() > 0);
        const bool need_accumulated = need_accumulated_for_detection || need_accumulated_for_publish;
        const bool publish_accumulated =
            need_accumulated_for_publish && ((cluster_tick_count_ % static_cast<uint64_t>(accumulated_publish_divider_)) == 0);

        if (!need_accumulated || input.cloud_queue.empty())
        {
            return input;
        }

        input.accumulated_cloud = boost::make_shared<PointCloud>();
        for (const auto& queued_cloud : input.cloud_queue)
        {
            if (queued_cloud && !queued_cloud->empty())
            {
                *input.accumulated_cloud += *queued_cloud;
            }
        }

        if (publish_accumulated && input.accumulated_cloud && !input.accumulated_cloud->empty())
        {
            const ros::Time accum_stamp = input.stamp.isZero() ? ros::Time::now() : input.stamp;
            visualizer_.publishCloud(input.accumulated_cloud, input.frame_id, accum_stamp, accumulated_cloud_pub_);
        }

        if (publish_raw_drone_cloud_)
        {
            input.raw_accumulated_cloud = boost::make_shared<PointCloud>();
            for (const auto& queued_cloud : input.raw_cloud_queue)
            {
                if (queued_cloud && !queued_cloud->empty())
                {
                    *input.raw_accumulated_cloud += *queued_cloud;
                }
            }
        }

        return input;
    }


    LidarDetector::DroneCandidate LidarDetector::findDrone(const DetectionInput& input)
    {
        if (track_enable_ && track_local_cluster_enable_ && tracker_.hasTrack())
        {
            DroneCandidate local_candidate = findDroneNearTrack(input);
            if (local_candidate.found)
            {
                return local_candidate;
            }
        }

        return detection_mode_ == DetectionMode::DYNAMIC ? findDynamicDrone(input) : findDirectDrone(input);
    }

    LidarDetector::DroneCandidate LidarDetector::findDroneNearTrack(const DetectionInput& input)
    {
        const PointCloudPtr source_cloud =
            detection_mode_ == DetectionMode::DIRECT ? input.accumulated_cloud :
            (!input.cloud_queue.empty() ? input.cloud_queue.back() : PointCloudPtr());
        const PointCloudPtr raw_source_cloud =
            detection_mode_ == DetectionMode::DIRECT ? input.raw_accumulated_cloud :
            (!input.raw_cloud_queue.empty() ? input.raw_cloud_queue.back() : PointCloudPtr());
        if (!source_cloud || source_cloud->empty())
        {
            return DroneCandidate();
        }

        const Eigen::Vector3f track_position = tracker_.getState().position;
        const float gate = std::max(0.05f, tracker_.getParams().gate_distance_m);
        const float roi_radius = std::max(gate, eps_ * 2.0f);
        const float roi_radius_sq = roi_radius * roi_radius;
        const float min_z = dynamic_detector_.getParams().min_z;

        std::vector<ClusterPoint> local_points;
        local_points.reserve(source_cloud->size());
        for (const auto& point : source_cloud->points)
        {
            if (point.z < min_z)
            {
                continue;
            }
            const float dx = point.x - track_position.x();
            const float dy = point.y - track_position.y();
            const float dz = point.z - track_position.z();
            if (dx * dx + dy * dy + dz * dz <= roi_radius_sq)
            {
                local_points.emplace_back(point.x, point.y, point.z);
            }
        }

        if (local_points.size() < static_cast<std::size_t>(minPts_))
        {
            return DroneCandidate();
        }

        DroneCandidate candidate = findBestClusterFromPoints(local_points, raw_source_cloud);
        if (!candidate.found)
        {
            return candidate;
        }

        const Eigen::Vector3f centroid(candidate.centroid.x(), candidate.centroid.y(), candidate.centroid.z());
        const float distance_to_track = (centroid - track_position).norm();
        if (distance_to_track > gate)
        {
            candidate.found = false;
            return candidate;
        }

        if (verbose_log_)
        {
            NODELET_INFO_THROTTLE(1.0,
                                  "LOCAL: points=%zu clusters=%zu dist=%.2f gate=%.2f",
                                  local_points.size(), candidate.clusters.size(), distance_to_track, gate);
        }
        return candidate;
    }

    LidarDetector::DroneCandidate LidarDetector::findBestClusterFromPoints(const std::vector<ClusterPoint>& points,
                                                                           const PointCloudPtr& raw_source_cloud)
    {
        DroneCandidate candidate;
        if (points.size() < static_cast<std::size_t>(minPts_))
        {
            return candidate;
        }

        std::vector<ClusterPoint> cluster_points = points;
        DBSCAN dbscan(cluster_points, eps_, minPts_);
        const int num_clusters = dbscan.run();
        if (num_clusters == 0)
        {
            return candidate;
        }

        candidate.clusters = dbscan.getClusters();
        candidate.cluster = cluster_filter_.findBest(candidate.clusters, candidate.bbox, candidate.centroid, &candidate.debug_infos);
        candidate.raw_source_cloud = raw_source_cloud;
        candidate.evaluated = true;
        candidate.found = candidate.cluster && !candidate.cluster->empty() && candidate.bbox.valid;
        return candidate;
    }

    LidarDetector::DroneCandidate LidarDetector::findDynamicDrone(const DetectionInput& input)
    {
        DroneCandidate candidate;
        if (input.cloud_queue.size() < frame_gap_ + 1)
        {
            if (verbose_log_)
            {
                NODELET_DEBUG_THROTTLE(1, "DYNAMIC: Waiting for more frames (%zu/%zu)",
                                    input.cloud_queue.size(), frame_gap_ + 1);
            }
            return candidate;
        }

        const auto current_cloud = input.cloud_queue.back();
        if (publish_raw_drone_cloud_ && !input.raw_cloud_queue.empty())
        {
            candidate.raw_source_cloud = input.raw_cloud_queue.back();
        }
        auto reference_it = input.cloud_queue.begin();
        std::advance(reference_it, input.cloud_queue.size() - frame_gap_ - 1);
        const auto reference_cloud = *reference_it;

        if (!current_cloud || current_cloud->empty() || !reference_cloud || reference_cloud->empty())
        {
            return candidate;
        }

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(reference_cloud);

        DynamicDetectorParams params = dynamic_detector_.getParams();
        dynamic_points_buffer_.clear();
        dynamic_points_buffer_.reserve(current_cloud->size());
        nn_indices_buffer_.clear();
        nn_distances_buffer_.clear();
        for (const auto& point : current_cloud->points)
        {
            if (point.z < params.min_z)
            {
                continue;
            }

            nn_indices_buffer_.clear();
            nn_distances_buffer_.clear();
            if (kdtree.radiusSearch(point, params.distance_threshold, nn_indices_buffer_, nn_distances_buffer_) == 0)
            {
                dynamic_points_buffer_.emplace_back(point.x, point.y, point.z);
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
            return candidate;
        }

        candidate = findBestClusterFromPoints(dynamic_points_buffer_, candidate.raw_source_cloud);
        if (verbose_log_ && candidate.evaluated)
        {
            NODELET_INFO("DYNAMIC: Found %zu clusters from %zu dynamic points", candidate.clusters.size(), dynamic_points_buffer_.size());
        }
        return candidate;
    }

    LidarDetector::DroneCandidate LidarDetector::findDirectDrone(const DetectionInput& input)
    {
        DroneCandidate candidate;
        if (!input.accumulated_cloud || input.accumulated_cloud->empty())
        {
            return candidate;
        }

        DynamicDetectorParams params = dynamic_detector_.getParams();
        candidate.raw_source_cloud = input.raw_accumulated_cloud;
        airspace_points_buffer_.clear();
        airspace_points_buffer_.reserve(input.accumulated_cloud->size());

        for (const auto& point : input.accumulated_cloud->points)
        {
            if (point.z >= params.min_z)
            {
                airspace_points_buffer_.emplace_back(point.x, point.y, point.z);
            }
        }

        if (verbose_log_)
        {
            NODELET_INFO_THROTTLE(1, "DIRECT: accumulated=%zu, airspace=%zu (min_z=%.1f)",
                                input.accumulated_cloud->size(), airspace_points_buffer_.size(), params.min_z);
        }

        if (airspace_points_buffer_.size() < static_cast<size_t>(minPts_))
        {
            return candidate;
        }

        candidate = findBestClusterFromPoints(airspace_points_buffer_, candidate.raw_source_cloud);
        if (verbose_log_ && candidate.evaluated)
        {
            NODELET_INFO("DIRECT: Found %zu clusters from %zu airspace points", candidate.clusters.size(), airspace_points_buffer_.size());
        }
        return candidate;
    }

    void LidarDetector::handleDroneCandidate(DroneCandidate& candidate,
                                            const std::string& frame_id,
                                            const ros::Time& stamp)
    {
        if (!candidate.evaluated)
        {
            return;
        }

        if (publish_cluster_debug_markers_)
        {
            publishClusterDebugMarkers(candidate.debug_infos, frame_id, stamp);
        }

        if (!candidate.found)
        {
            cloud_processor_.resetAabbLocalAccumulation();
            if (track_enable_)
            {
                tracker_.markMiss(stamp);
                publishTrackState(stamp, nullptr);
            }
            return;
        }

        auto detected_cluster = candidate.cluster;
        BBox3D detected_bbox = candidate.bbox;
        Eigen::Vector4f detected_centroid = candidate.centroid;
        ClusterDebugInfo best_info;
        double second_score = -1.0;

        CandidateSelectionContext selection_context;
        if (track_enable_ && tracker_.hasTrack())
        {
            selection_context.use_track_position = true;
            selection_context.track_position = tracker_.getState().position;
            selection_context.track_gate_distance = tracker_.getParams().gate_distance_m;
        }

        const CandidateSelectionResult selection_result =
            candidate_selector_.select(candidate.clusters, candidate.debug_infos, selection_context);
        const bool has_best_info = selection_result.found;
        if (has_best_info)
        {
            best_info = selection_result.info;
            second_score = selection_result.second_score;
            if (best_info.cluster_index >= 0 && best_info.cluster_index < static_cast<int>(candidate.clusters.size()) &&
                candidate.clusters[best_info.cluster_index])
            {
                detected_cluster = candidate.clusters[best_info.cluster_index];
                detected_bbox = best_info.bbox;
                detected_centroid = best_info.centroid;
            }
        }
        double confidence = has_best_info ? best_info.score : 1.0;
        candidate.raw_drone_cloud = cropRawDroneCloud(candidate.raw_source_cloud, detected_bbox);
        candidate.debug_drone_cloud = publish_raw_drone_cloud_ && candidate.raw_drone_cloud && !candidate.raw_drone_cloud->empty()
                                           ? candidate.raw_drone_cloud
                                           : detected_cluster;
        candidate.target_source_drone_cloud.reset();
        if (candidate.raw_drone_cloud && !candidate.raw_drone_cloud->empty())
        {
            candidate.target_source_drone_cloud = cloud_processor_.updateAabbLocalAccumulation(
                candidate.raw_drone_cloud, detected_bbox.center, drone_aabb_accumulation_max_points_);
        }

        if (!track_enable_)
        {
            if (candidate.target_source_drone_cloud && !candidate.target_source_drone_cloud->empty())
            {
                findTarget(candidate.target_source_drone_cloud, frame_id, stamp);
            }
            publishResults(detected_cluster, detected_bbox, detected_centroid, confidence, frame_id, stamp,
                        candidate.debug_drone_cloud, candidate.target_source_drone_cloud);
            return;
        }

        const Eigen::Vector3f measurement(detected_centroid.x(), detected_centroid.y(), detected_centroid.z());
        const auto& tracker_state = tracker_.getState();
        double dt = 0.1;
        if (tracker_state.initialized && !tracker_state.stamp.isZero() && stamp > tracker_state.stamp)
        {
            dt = (stamp - tracker_state.stamp).toSec();
        }

        SingleTargetTracker::AssociationResult assoc_result;
        if (has_best_info)
        {
            assoc_result = tracker_.updateAssociation(&measurement,
                                                    best_info.ratio,
                                                    best_info.volume,
                                                    best_info.pca_ratio,
                                                    best_info.score,
                                                    second_score,
                                                    static_cast<double>(measurement.norm()),
                                                    dt);
            confidence = assoc_result.confidence;
        }
        else
        {
            assoc_result.accepted = true;
            assoc_result.locked = true;
            assoc_result.confidence = confidence;
        }

        if (!assoc_result.accepted)
        {
            tracker_.markMiss(stamp);
            publishTrackState(stamp, nullptr);
            return;
        }

        const bool updated = tracker_.updateMeasurement(measurement, stamp);
        if (!updated)
        {
            tracker_.markMiss(stamp);
            if (!tracker_.hasTrack())
            {
                tracker_.reset();
                tracker_.updateMeasurement(measurement, stamp);
            }
        }

        tracker_.updateAabbPoints(detected_bbox);
        if (candidate.target_source_drone_cloud && !candidate.target_source_drone_cloud->empty())
        {
            findTarget(candidate.target_source_drone_cloud, frame_id, stamp);
        }
        publishResults(detected_cluster, detected_bbox, detected_centroid, confidence, frame_id, stamp,
                    candidate.debug_drone_cloud, candidate.target_source_drone_cloud);
        publishTrackState(stamp, &detected_bbox, detected_cluster);
    }

    // 4. 找 target：从 drone/输入点云中提取 target cloud，并交给 recognition 找 target 点。

    void LidarDetector::findTarget(const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_source_drone_cloud,
                                            const std::string& frame_id,
                                            const ros::Time& stamp)
        {
            const auto target_cloud = target_cloud_extractor_.extract(target_source_drone_cloud);
            if (!target_cloud || target_cloud->empty())
            {
                return;
            }

            visualizer_.publishCloud(target_cloud, frame_id, stamp, target_cloud_pub_);
            const auto front_view_output = target_front_view_controller_.show(target_cloud, frame_id, stamp);
            if (front_view_output.projected_surface_cloud && !front_view_output.projected_surface_cloud->empty())
            {
                visualizer_.publishCloud(front_view_output.projected_surface_cloud,
                                        frame_id,
                                        stamp,
                                        target_projected_surface_cloud_pub_);
            }

            pcl::PointXYZ target_min(std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max());
            pcl::PointXYZ target_max(std::numeric_limits<float>::lowest(),
                                    std::numeric_limits<float>::lowest(),
                                    std::numeric_limits<float>::lowest());
            for (const auto& point : target_cloud->points)
            {
                target_min.x = std::min(target_min.x, point.x);
                target_min.y = std::min(target_min.y, point.y);
                target_min.z = std::min(target_min.z, point.z);
                target_max.x = std::max(target_max.x, point.x);
                target_max.y = std::max(target_max.y, point.y);
                target_max.z = std::max(target_max.z, point.z);
            }
            auto target_marker = Visualizer::createAABBMarker(
                target_min, target_max, 101, "target_cloud", frame_id, Eigen::Vector4d(0.0, 1.0, 1.0, 0.6), 1.0);
            target_marker.header.stamp = stamp;
            target_marker_pub_.publish(target_marker);

        }

    // 5. 发布：所有 ROS 输出集中在这里。

    void LidarDetector::publishResults(const pcl::PointCloud<pcl::PointXYZ>::Ptr& detected_cluster,
                                        const BBox3D& detected_bbox,
                                        const Eigen::Vector4f& detected_centroid,
                                        double confidence,
                                        const std::string& frame_id,
                                        const ros::Time& stamp,
                                        const PointCloudPtr& debug_drone_cloud,
                                        const PointCloudPtr& target_source_drone_cloud)
        {
            if (detected_cluster && !detected_cluster->empty()) {
                if (debug_drone_cloud && !debug_drone_cloud->empty())
                {
                    visualizer_.publishCloud(debug_drone_cloud, frame_id, stamp, drone_cloud_pub_);
                }
                if (target_source_drone_cloud && !target_source_drone_cloud->empty())
                {
                    visualizer_.publishCloud(target_source_drone_cloud, frame_id, stamp, drone_aabb_accumulated_cloud_pub_);
                }
                
                auto marker = Visualizer::createOBBMarker(
                    detected_cluster, 0, frame_id,
                    Eigen::Vector4d(1.0, 0.0, 0.0, 0.5));
                marker.header.stamp = stamp;
                marker.ns = "detected_cluster_obb";
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
                detection_msg.confidence = clamp01(confidence);

                Eigen::Vector3f drone_position(detected_centroid.x(), detected_centroid.y(), detected_centroid.z());
                if (track_enable_ && tracker_.hasTrack())
                {
                    drone_position = tracker_.getState().position;
                }

                detection_msg.drone_position.x = drone_position.x();
                detection_msg.drone_position.y = drone_position.y();
                detection_msg.drone_position.z = drone_position.z();
                detection_msg.target_position_valid = getLatestTargetPoint(frame_id, stamp, detection_msg.target_position);

                // Compatibility only: old consumers still compile, but LiDAR semantics use drone_position/target_position.
                detection_msg.pose.orientation.w = 1.0;

                detection_pub_.publish(detection_msg);

                NODELET_DEBUG("Published detection: drone=(%.2f, %.2f, %.2f), target_valid=%d",
                            drone_position.x(), drone_position.y(), drone_position.z(),
                            static_cast<int>(detection_msg.target_position_valid));
            }
        }

    pcl::PointCloud<pcl::PointXYZ>::Ptr LidarDetector::cropRawDroneCloud(
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_source_cloud,
            const BBox3D& detected_bbox)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr raw_drone_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (!raw_source_cloud || raw_source_cloud->empty() || !detected_bbox.valid)
            {
                return raw_drone_cloud;
            }

            const pcl::PointXYZ min_pt(detected_bbox.min_pt.x(), detected_bbox.min_pt.y(), detected_bbox.min_pt.z());
            const pcl::PointXYZ max_pt(detected_bbox.max_pt.x(), detected_bbox.max_pt.y(), detected_bbox.max_pt.z());
            const auto& filter_params = cloud_processor_.getParams();
            const float expansion = filter_params.enable_voxel_downsample ? filter_params.voxel_leaf : 0.0f;
            raw_drone_cloud = cloud_processor_.extractInAABB(raw_source_cloud, min_pt, max_pt, expansion);
            raw_drone_cloud->header = raw_source_cloud->header;
            raw_drone_cloud->width = static_cast<uint32_t>(raw_drone_cloud->size());
            raw_drone_cloud->height = 1;
            raw_drone_cloud->is_dense = false;
            return raw_drone_cloud;
        }

    void LidarDetector::publishSelectedTargetPoint(const pcl::PointXYZ& point,
                                                    const std::string& frame_id,
                                                    const ros::Time& stamp)
        {
            const std::string output_frame_id = frame_id.empty() ? frame_id_ : frame_id;
            const ros::Time output_stamp = stamp.isZero() ? ros::Time::now() : stamp;

            geometry_msgs::Point stable_point;
            stable_point.x = point.x;
            stable_point.y = point.y;
            stable_point.z = point.z;
            bool update_cache = true;

            {
                std::lock_guard<std::mutex> lock(target_point_mutex_);
                if (has_latest_target_point_ && latest_target_point_frame_id_ == output_frame_id)
                {
                    const double dx = point.x - latest_target_point_.x;
                    const double dy = point.y - latest_target_point_.y;
                    const double dz = point.z - latest_target_point_.z;
                    const double jump_distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                    const bool recent_point = latest_target_point_stamp_.isZero() ||
                                              output_stamp.isZero() ||
                                              std::abs((output_stamp - latest_target_point_stamp_).toSec()) <= 0.5;
                    const bool jump_too_large = recent_point && target_point_max_jump_distance_ > 0.0 &&
                                                jump_distance > target_point_max_jump_distance_;
                    if (jump_too_large)
                    {
                        stable_point = latest_target_point_;
                        update_cache = false;
                    }
                    else if (recent_point)
                    {
                        const double alpha = std::clamp(target_point_smoothing_alpha_, 0.0, 1.0);
                        stable_point.x = latest_target_point_.x + alpha * dx;
                        stable_point.y = latest_target_point_.y + alpha * dy;
                        stable_point.z = latest_target_point_.z + alpha * dz;
                    }
                }

                if (update_cache)
                {
                    latest_target_point_ = stable_point;
                    latest_target_point_frame_id_ = output_frame_id;
                    latest_target_point_stamp_ = output_stamp;
                    has_latest_target_point_ = true;
                }
            }

            visualization_msgs::Marker marker;
            marker.header.frame_id = output_frame_id;
            marker.header.stamp = output_stamp;
            marker.ns = "selected_target_point";
            marker.id = 0;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position = stable_point;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = selected_target_point_marker_scale_;
            marker.scale.y = selected_target_point_marker_scale_;
            marker.scale.z = selected_target_point_marker_scale_;
            marker.color.r = 1.0;
            marker.color.g = 0.1;
            marker.color.b = 0.0;
            marker.color.a = 1.0;
            marker.lifetime = ros::Duration(0.0);
            selected_target_point_pub_.publish(marker);
        }

    bool LidarDetector::getLatestTargetPoint(const std::string& frame_id,
                                             const ros::Time& stamp,
                                             geometry_msgs::Point& target_point) const
        {
            std::lock_guard<std::mutex> lock(target_point_mutex_);
            if (!has_latest_target_point_ || latest_target_point_frame_id_ != frame_id)
            {
                return false;
            }
            if (!stamp.isZero() && !latest_target_point_stamp_.isZero() &&
                std::abs((stamp - latest_target_point_stamp_).toSec()) > 0.5)
            {
                return false;
            }
            target_point = latest_target_point_;
            return true;
        }

    void LidarDetector::publishTrackState(const ros::Time& stamp,
                                            const BBox3D* bbox_hint,
                                            const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& track_cloud)
        {
            if (bbox_hint)
            {
                tracker_.updateAabbPoints(*bbox_hint);
            }

            const auto& state = tracker_.getState();
            const bool has_track = tracker_.hasTrack();
            const std::string frame_id = track_frame_id_.empty() ? frame_id_ : track_frame_id_;

            rm_radar_msgs::DroneTrackData track_msg;
            track_msg.header.stamp = stamp;
            track_msg.header.frame_id = frame_id;
            if (track_cloud && !track_cloud->empty())
            {
                pcl::toROSMsg(*track_cloud, track_msg.track_cloud);
                track_msg.track_cloud.header.stamp = stamp;
                track_msg.track_cloud.header.frame_id = frame_id;
            }
            track_msg.tracking = has_track;
            track_msg.position.x = state.position.x();
            track_msg.position.y = state.position.y();
            track_msg.position.z = state.position.z();
            track_msg.yaw = state.yaw;
            track_msg.velocity.x = state.velocity.x();
            track_msg.velocity.y = state.velocity.y();
            track_msg.velocity.z = state.velocity.z();
            track_msg.v_yaw = state.v_yaw;
            track_msg.accel = state.accel;

            const double half_yaw = static_cast<double>(state.yaw) * 0.5;
            track_msg.rotation.w = std::cos(half_yaw);
            track_msg.rotation.x = 0.0;
            track_msg.rotation.y = 0.0;
            track_msg.rotation.z = std::sin(half_yaw);

            if (bbox_hint && bbox_hint->valid)
            {
                track_msg.radius = 0.5 * std::max(static_cast<double>(bbox_hint->dimensions.x()),
                                                static_cast<double>(bbox_hint->dimensions.y()));
                track_msg.dz = bbox_hint->dimensions.z();
            }

            if (tracker_.hasAabbPoints())
            {
                const auto& aabb_points = tracker_.aabbPoints();
                for (size_t point_index = 0; point_index < aabb_points.size(); ++point_index)
                {
                    track_msg.aabb_points[point_index] = aabb_points[point_index];
                }
            }

            track_pub_.publish(track_msg);

            if (publish_track_target_marker_)
            {
                track_marker_pub_.publish(Visualizer::createTrackTargetSphereMarker(
                    state, has_track, frame_id, 0, stamp, 0.2, 0.4));
                track_marker_pub_.publish(Visualizer::createTrackVelocityArrowMarker(
                    state, has_track, frame_id, 1, stamp, 0.2, 0.05, 0.1, 0.2, 0.05, 0.25, 3.0));
            }
            else
            {
                track_marker_pub_.publish(Visualizer::createDeleteMarker(frame_id, "track_target", 0, stamp));
                track_marker_pub_.publish(Visualizer::createDeleteMarker(frame_id, "track_target", 1, stamp));
            }

            track_marker_pub_.publish(Visualizer::createTrackTrajectoryMarker(
                state, has_track, publish_track_trajectory_, frame_id, 2, stamp, 0.05, 0.05f, 80, track_trajectory_points_));

            if (publish_track_text_)
            {
                track_marker_pub_.publish(Visualizer::createTrackSpeedTextMarker(
                    state, has_track, frame_id, 3, stamp, 0.2, 0.6f, 0.35f));
            }
            else
            {
                track_marker_pub_.publish(Visualizer::createDeleteMarker(frame_id, "track_text", 3, stamp));
            }

            if (has_track)
            {
                const Eigen::Quaternionf orientation(Eigen::AngleAxisf(state.yaw, Eigen::Vector3f::UnitZ()));
                Visualizer::broadcastTF(state.position, orientation, frame_id, "tracked_target_0", tf_broadcaster_);
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
