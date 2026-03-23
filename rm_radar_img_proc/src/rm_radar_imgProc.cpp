#include <rm_radar_imgProc.h>
#include <iomanip>

PLUGINLIB_EXPORT_CLASS(rm_radarplugin::Processor, nodelet::Nodelet);

namespace rm_radarplugin
{
    void Processor::onInit()
    {
    ros::NodeHandle& nh = getMTPrivateNodeHandle();
    initialize(nh);
    static ros::CallbackQueue my_queue;
    nh.setCallbackQueue(&my_queue);
    this->my_thread_ = std::thread([]() {
        ros::SingleThreadedSpinner spinner;
        spinner.spin(&my_queue);
    });
    }

    void Processor::initialize(ros::NodeHandle &nh)
    {
        nh_ = ros::NodeHandle(nh, "radar_imgProc");
        ROS_INFO("radar image process initialized");

        auto armor_params_init = [this, &nh]() {
            ROS_INFO("reading armor param");
            //    bar_br_thresh_ = nh.param("bar_br_thresh", decltype(bar_br_thresh_){});
            /// using to define 'Point2d' in id-classification
            bar_length_in_warp_ = nh.param("bar_length_in_warp", decltype(bar_length_in_warp_){});
            warp_height_ = nh.param("warp_height", decltype(warp_height_){});
            top_light_y_ = nh.param("top_light_y", decltype(top_light_y_){});
            roi_width_ = nh.param("roi_width", decltype(roi_width_){});
            roi_height_ = nh.param("roi_height", decltype(roi_height_){});

            is_bar_debug_ = nh.param("is_bar_debug", decltype(is_bar_debug_){});
            is_armor_debug_ = nh.param("is_armor_debug", decltype(is_armor_debug_){});
            select_bar_ = nh.param("select_bar", decltype(select_bar_){});
            max_angle_diff_ = nh.param("max_angle_diff", decltype(max_angle_diff_){});
            min_lw_ratio_ = nh.param("min_lw_ratio", decltype(min_lw_ratio_){});
            max_lw_ratio_ = nh.param("max_lw_ratio", decltype(max_lw_ratio_){});
            min_pixel_contained_ratio_ = nh.param("min_pixel_contained_ratio", decltype(min_pixel_contained_ratio_){});
            max_bars_ratio_ = nh.param("max_bars_ratio", decltype(max_bars_ratio_){});
            max_bars_distance_ = nh.param("max_bars_distance", decltype(max_bars_distance_){});
            min_bars_distance_ = nh.param("min_bars_distance", decltype(min_bars_distance_){});
            max_bars_angle_ = nh.param("max_bars_angle", decltype(max_bars_angle_){});
            max_bars_x_dis_ = nh.param("max_bars_x_dis", decltype(max_bars_x_dis_){});
            select_by_last_ = nh.param("select_by_last", decltype(select_by_last_){});

            warp_thresh_ = nh.param("warp_thresh", decltype(warp_thresh_){});
            gamma_ = nh.param("gamma", decltype(gamma_){});
            contrast_alpha_ = nh.param("contrast_alpha", decltype(contrast_alpha_){});
            contrast_beta_ = nh.param("contrast_beta", decltype(contrast_beta_){});
            gamma_y_ = nh.param("gamma_y", decltype(gamma_y_){});
            rotate_ = nh.param("rotate", decltype(rotate_){});

            expand_ratio_ = nh.param("expand_ratio", decltype(expand_ratio_){});
            id_confidence_ = nh.param("id_confidence", decltype(id_confidence_){});
            input_shape_ = nh.param("input_shape", decltype(input_shape_){});
            use_id_cls_ = nh.param("use_id_cls", decltype(use_id_cls_){});
            min_id_white_ratio_ = nh.param("min_id_white_ratio", decltype(min_id_white_ratio_){});
            max_id_white_ratio_ = nh.param("max_id_white_ratio", decltype(max_id_white_ratio_){});

            image_x_center_ = nh.param("image_x_center", decltype(image_x_center_){});
            image_y_center_ = nh.param("image_y_center", decltype(image_y_center_){});

            camera_coordinate_ = nh.param("camera_coordinate", decltype(camera_coordinate_){});
            ROS_INFO("Armor params reading done");
        };
        auto pre_process_params_init = [this, &nh]() {
            ROS_INFO("reading pre-process param");
            target_is_red_ = nh.param("target_color", decltype(target_is_red_){});
            preprocess_method_ = nh.param("preprocess_method", decltype(preprocess_method_){});

            red_h_min_low_ = nh.param("red_h_min_low", decltype(red_h_min_low_){});
            red_h_max_low_ = nh.param("red_h_max_low", decltype(red_h_max_low_){});
            red_h_min_high_ = nh.param("red_h_min_high", decltype(red_h_min_high_){});
            red_h_max_high_ = nh.param("red_h_max_high", decltype(red_h_max_high_){});
            red_s_min_ = nh.param("red_s_min", decltype(red_s_min_){});
            red_s_max_ = nh.param("red_s_max", decltype(red_s_max_){});
            red_v_min_ = nh.param("red_v_min", decltype(red_v_min_){});
            red_v_max_ = nh.param("red_v_max", decltype(red_v_max_){});

            blue_h_min_ = nh.param("blue_h_min", decltype(blue_h_min_){});
            blue_h_max_ = nh.param("blue_h_max", decltype(blue_h_max_){});
            blue_s_min_ = nh.param("blue_s_min", decltype(blue_s_min_){});
            blue_s_max_ = nh.param("blue_s_max", decltype(blue_s_max_){});
            blue_v_min_ = nh.param("blue_v_min", decltype(blue_v_min_){});
            blue_v_max_ = nh.param("blue_v_max", decltype(blue_v_max_){});

            binary_thresh_ = nh.param("binary_thresh", decltype(binary_thresh_){});
            morph_type_ = nh.param("morph_type", decltype(morph_type_){});
            binary_element_ = nh.param("binary_element", decltype(binary_element_){});
            ROS_INFO("pre-processing param reading done");
        };

        armor_params_init();
        pre_process_params_init();

        armor_cfg_srv_ = new dynamic_reconfigure::Server<rm_radar_img_proc::ArmorConfig>(ros::NodeHandle(nh_, "armor_condition"));
        armor_cfg_cb_ = boost::bind(&Processor::armorconfigCB, this, _1, _2);
        armor_cfg_srv_->setCallback(armor_cfg_cb_);

        preprocess_cfg_srv_ = new dynamic_reconfigure::Server<rm_radar_img_proc::PreprocessConfig>(ros::NodeHandle(nh_, "preprocess_condition"));
        preprocess_cfg_cb_ = boost::bind(&Processor::preProcessconfigCB, this, _1, _2);
        preprocess_cfg_srv_->setCallback(preprocess_cfg_cb_);

        draw_cfg_srv_ = new dynamic_reconfigure::Server<rm_radar_img_proc::DrawConfig>(ros::NodeHandle(nh_, "draw_condition"));
        draw_cfg_cb_ = boost::bind(&Processor::drawconfigCB, this, _1, _2);
        draw_cfg_srv_->setCallback(draw_cfg_cb_);

        it_ = std::make_shared<image_transport::ImageTransport>(nh_);
        image_pub_ = it_->advertise("debug_image", 1);

        //cam_sub
        tele_cam_sub_ = it_->subscribeCamera("/hk_camera/image_raw", 1, &Processor::tele_cam_callback, this);

        //tracker_sub
        track_sub_ = nh.subscribe("/tracker/track_data", 1, &Processor::trackerCB, this);

        tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(ros::Duration(10));
        tf_listener_ = new tf2_ros::TransformListener(*tf2_buffer_);
        target_pub_ = nh.advertise<decltype(target_array_)>("/processor/result_msg", 1);
        target_pub_single_ = nh.advertise<rm_radar_msgs::DroneDetection>("/processor/single_result_msg", 1);
        //tracker data pub
        //wide_cam_sub_ = it_->subscribeCamera("/hk_camera/image_raw", 20, &Processor::wide_cam_callback, this);


    }

    //dynamic reconfigure callback
    void Processor::armorconfigCB(rm_radar_img_proc::ArmorConfig& config, uint32_t level)
    {
        if (!armor_dynamic_reconfig_initialized_)
        {
            is_bar_debug_ = config.is_bar_debug;
            is_armor_debug_ = config.is_armor_debug;
            select_bar_ = config.select_bar;
            max_angle_diff_ = config.max_angle_diff;
            min_lw_ratio_ = config.min_lw_ratio;
            max_lw_ratio_ = config.max_lw_ratio;
            min_pixel_contained_ratio_ = config.min_pixel_contained_ratio;
            max_bars_ratio_ = config.max_bars_ratio;
            max_bars_distance_ = config.max_bars_distance;
            min_bars_distance_ = config.min_bars_distance;
            max_bars_angle_ = config.max_bars_angle;
            max_bars_x_dis_ = config.max_bars_x_dis;
            select_by_last_ = config.select_by_last;

            warp_thresh_ = config.warp_thresh;
            gamma_ = config.gamma;
            contrast_alpha_ = config.contrast_alpha;
            contrast_beta_ = config.contrast_beta;
            gamma_y_ = config.gamma_y;
            rotate_ = config.rotate;

            id_confidence_ = config.id_confidence;
            use_id_cls_ = config.use_id_cls;
            min_id_white_ratio_ = config.min_id_white_ratio;
            max_id_white_ratio_ = config.max_id_white_ratio;

            armor_dynamic_reconfig_initialized_ = true;
        }

        ROS_INFO("reading Armor dynamic reconfigure callback");
        is_bar_debug_ = config.is_bar_debug;
        is_armor_debug_ = config.is_armor_debug;
        select_bar_ = config.select_bar;
        max_angle_diff_ = config.max_angle_diff;
        min_lw_ratio_ = config.min_lw_ratio;
        max_lw_ratio_ = config.max_lw_ratio;
        min_pixel_contained_ratio_ = config.min_pixel_contained_ratio;
        max_bars_ratio_ = config.max_bars_ratio;
        max_bars_distance_ = config.max_bars_distance;
        min_bars_distance_ = config.min_bars_distance;
        max_bars_angle_ = config.max_bars_angle;
        max_bars_x_dis_ = config.max_bars_x_dis;
        select_by_last_ = config.select_by_last;

        warp_thresh_ = config.warp_thresh;
        gamma_ = config.gamma;
        contrast_alpha_ = config.contrast_alpha;
        contrast_beta_ = config.contrast_beta;
        gamma_y_ = config.gamma_y;
        rotate_ = config.rotate;

        id_confidence_ = config.id_confidence;
        use_id_cls_ = config.use_id_cls;
        min_id_white_ratio_ = config.min_id_white_ratio;
        max_id_white_ratio_ = config.max_id_white_ratio;

        uchar* p = look_up_table_.ptr();
        for (int i = 0; i < 256; ++i)
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_y_) * 255.0);

        ROS_INFO("Armor dynamic reconfigure callback done");
    }


    void Processor::preProcessconfigCB(rm_radar_img_proc::PreprocessConfig& config, uint32_t level)
    {
        if (!pre_process_dynamic_reconfig_initialized_)
        {
            target_is_red_ = config.target_color;
            preprocess_method_ = config.preprocess_method;

            red_h_min_low_ = config.red_h_min_low;
            red_h_max_low_ = config.red_h_max_low;
            red_h_min_high_ = config.red_h_min_high;
            red_h_max_high_ = config.red_h_max_high;
            red_s_min_ = config.red_s_min;
            red_s_max_ = config.red_s_max;
            red_v_min_ = config.red_v_min;
            red_v_max_ = config.red_v_max;

            blue_h_min_ = config.blue_h_min;
            blue_h_max_ = config.blue_h_max;
            blue_s_min_ = config.blue_s_min;
            blue_s_max_ = config.blue_s_max;
            blue_v_min_ = config.blue_v_min;
            blue_v_max_ = config.blue_v_max;

            binary_thresh_ = config.binary_thresh;

            morph_type_ = config.morph_type;
            binary_element_ = config.binary_element;

            pre_process_dynamic_reconfig_initialized_ = true;
        }

        target_is_red_ = config.target_color;
        preprocess_method_ = config.preprocess_method;
        red_h_min_low_ = config.red_h_min_low;
        red_h_max_low_ = config.red_h_max_low;
        red_h_min_high_ = config.red_h_min_high;
        red_h_max_high_ = config.red_h_max_high;
        red_s_min_ = config.red_s_min;
        red_s_max_ = config.red_s_max;
        red_v_min_ = config.red_v_min;
        red_v_max_ = config.red_v_max;
        blue_h_min_ = config.blue_h_min;
        blue_h_max_ = config.blue_h_max;
        blue_s_min_ = config.blue_s_min;
        blue_s_max_ = config.blue_s_max;
        blue_v_min_ = config.blue_v_min;
        blue_v_max_ = config.blue_v_max;
        binary_thresh_ = config.binary_thresh;
        morph_type_ = config.morph_type;
        binary_element_ = config.binary_element;
        if (binary_element_ % 2 == 0) binary_element_ += 1; 
    }


    void Processor::drawBars(cv::Mat& image)
    {
    cv::Scalar line_color = cv::Scalar(0, 255, 0);
    for (const auto& bar : bars_)
        for (int j = 0; j < 4; j++)
        cv::line(image, bar.points_[j], bar.points_[(j + 1) % 4], line_color, line_width_);
    }

    void Processor::drawArmors(cv::Mat& image, std::vector<Armor>& armors)
    {
    cv::Scalar line_color = cv::Scalar(0, 255, 0);
    for (const auto& armor : armors)
    {
        for (int j = 0; j < 4; j++)
            cv::line(image, armor.bars_4points_[j], armor.bars_4points_[(j + 1) % 4], line_color, line_width_);

        // Debug: score + angle (PCA-major-axis)
        {
            std::ostringstream ss1;
            ss1 << std::fixed << std::setprecision(2) << "s:" << armor.confidence_;
            cv::putText(image, ss1.str(), armor.bars_4points_[0], cv::FONT_HERSHEY_COMPLEX,
                        1.0, cv::Scalar(0, 255, 0), 2);

            // std::ostringstream ss2;
            // ss2 << std::fixed << std::setprecision(1) << "a:" << armor.getArmorAnglePcaDeg();
            // cv::Point2d p = armor.bars_4points_[0];
            // p.y -= 50;
            // cv::putText(image, ss2.str(), p, cv::FONT_HERSHEY_COMPLEX,
            //             1.0, cv::Scalar(0, 255, 255), 2);
        }
    }
    }
    void Processor::drawArmorsVertexes(cv::Mat& image, std::vector<Armor>& armors)
    {
    cv::Scalar vertex_color = cv::Scalar(0, 255, 0);
    for (const auto& armor : armors)
    {
        for (int i = 0; i < 4; i++)
        {
        circle(image, armor.bars_4points_[i], 3, vertex_color, 2);
        putText(image, std::to_string(i), armor.bars_4points_[i], cv::FONT_HERSHEY_COMPLEX, 1.5, vertex_color);
        }
        circle(image, armor.center_, 3, vertex_color, 2);
        putText(image, std::to_string(5), armor.center_, cv::FONT_HERSHEY_COMPLEX, 1.5, vertex_color);

        line(image, armor.bars_4points_[0], armor.bars_4points_[1], cv::Scalar(0, 255, 0), 2, 8, 0);
        line(image, armor.bars_4points_[2], armor.bars_4points_[3], cv::Scalar(0, 255, 0), 2, 8, 0);
    }
    }

    bool Processor::drawWarp()
    {
        if (armors_.size() != 1)
            return false;
        Armor armor_tmp(armors_[0]);
        return true;
    }

    void Processor::trackerCB(const rm_msgs::TrackData::ConstPtr& msg)
    {
        // Store the latest track data
        track_data_ = *msg;
        ROS_INFO_ONCE("[Tracker] Received track data");
    }

    bool Processor::projectPoint3Dto2D(const geometry_msgs::Point& point_3d, cv::Point2d& point_2d)
    {
        if(intrinsics_.empty() || dist_coeffs_.empty())
        {
            ROS_WARN_THROTTLE(5, "[Project] Camera intrinsics or dist coeffs is empty");
            return false;
        }

        // Transform 3D point from odom frame to camera frame
        geometry_msgs::PointStamped point_in, point_out;
        point_in.header = track_data_.header;
        point_in.point = point_3d;

        try
        {
            // Transform from tracker frame (odom) to camera frame
            geometry_msgs::TransformStamped transform_stamped = 
                tf2_buffer_->lookupTransform(camera_info_->header.frame_id,  // target: camera frame
                                            track_data_.header.frame_id,     // source: odom frame
                                            ros::Time(0),                     // latest
                                            ros::Duration(0.1));
            tf2::doTransform(point_in, point_out, transform_stamped);
            
            // // Debug: Print transformation details
            // ROS_INFO_THROTTLE(1, "[Project] Transform: %s -> %s", 
            //     track_data_.header.frame_id.c_str(), 
            //     camera_info_->header.frame_id.c_str());
            // ROS_INFO_THROTTLE(1, "[Project] Odom pos: (%.3f, %.3f, %.3f) -> Camera pos: (%.3f, %.3f, %.3f)",
            //     point_in.point.x, point_in.point.y, point_in.point.z,
            //     point_out.point.x, point_out.point.y, point_out.point.z);
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN_THROTTLE(5, "[Project] Transform failure: %s", ex.what());
            return false;
        }

        // Now project 3D point (in camera frame) to 2D image
        std::vector<cv::Point3d> object_points;
        object_points.emplace_back(cv::Point3d(point_out.point.x, point_out.point.y, point_out.point.z));

        std::vector<cv::Point2d> image_points;
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);  // No rotation (already in camera frame)
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);  // No translation

        cv::projectPoints(object_points, rvec, tvec, intrinsics_, dist_coeffs_, image_points);

        if(!image_points.empty())
        {
            point_2d = image_points[0];
            // ROS_INFO_THROTTLE(1, "[Project] 2D projection: (%.1f, %.1f)", point_2d.x, point_2d.y);
            return true;
        }
        return false;
    }

    void Processor::drawTracker(cv::Mat& image)
    {
        // Draw detected armors
        drawArmorsVertexes(image, armors_);
        // drawArmors(image, armors_);

        // Draw detection info panel (top-left corner)
        int panel_y = 60;
        if(!armors_.empty())
        {   
            if(!target_array_.detections.empty())
            {
                auto& det = target_array_.detections[0];
                std::stringstream ss_det;
                ss_det << std::fixed << std::setprecision(3) 
                       << "Det pos: (" << det.pose.position.x << ", " 
                       << det.pose.position.y << ", " 
                       << det.pose.position.z << ")";
                cv::putText(image, ss_det.str(), cv::Point(10, panel_y),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
                panel_y += 20;
            }
        }
        
        // Draw tracker position if tracking
        if(track_data_.tracking)
        {
            // Draw tracker info panel
            cv::putText(image, "=== TRACKER (odom frame) ===", cv::Point(10, panel_y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            panel_y += 20;
            
            std::stringstream ss_track;
            ss_track << std::fixed << std::setprecision(3) 
                     << "Track pos: (" << track_data_.position.x << ", " 
                     << track_data_.position.y << ", " 
                     << track_data_.position.z << ")";
            cv::putText(image, ss_track.str(), cv::Point(10, panel_y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            panel_y += 20;
            
            ss_track.str("");
            ss_track << std::fixed << std::setprecision(3) 
                     << "Track vel: (" << track_data_.velocity.x << ", " 
                     << track_data_.velocity.y << ", " 
                     << track_data_.velocity.z << ")";
            cv::putText(image, ss_track.str(), cv::Point(10, panel_y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            panel_y += 20;

            // Calculate dt: time difference between tracker data and current image
            double dt = 0.0;
            if(!track_data_.header.stamp.isZero() && camera_info_)
            {
                dt = (camera_info_->header.stamp - track_data_.header.stamp).toSec();
                // Clamp dt to reasonable range
                if(dt < 0) dt = 0;
                if(dt > 0.5) dt = 0.5;  // Max 500ms delay
            }

            ss_track.str("");
            ss_track << "dt: " << std::fixed << std::setprecision(3) << dt << "s";
            cv::putText(image, ss_track.str(), cv::Point(10, panel_y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            panel_y += 20;

            // Compensate position using velocity and dt
            geometry_msgs::Point compensated_pos;
            compensated_pos.x = track_data_.position.x + track_data_.velocity.x * dt;
            compensated_pos.y = track_data_.position.y + track_data_.velocity.y * dt;
            compensated_pos.z = track_data_.position.z + track_data_.velocity.z * dt;

            cv::Point2d tracker_point_2d;
            if(projectPoint3Dto2D(compensated_pos, tracker_point_2d))
            {
                // Check if point is within image bounds
                if(tracker_point_2d.x >= 0 && tracker_point_2d.x < image.cols &&
                   tracker_point_2d.y >= 0 && tracker_point_2d.y < image.rows)
                {
                    // Draw tracker point (larger circle, different color)
                    cv::circle(image, tracker_point_2d, 15, cv::Scalar(0, 0, 255), 3);  // Red circle
                    cv::circle(image, tracker_point_2d, 5, cv::Scalar(0, 255, 255), -1); // Yellow center

                    // Draw crosshair
                    int crosshair_size = 20;
                    cv::line(image, 
                            cv::Point(tracker_point_2d.x - crosshair_size, tracker_point_2d.y),
                            cv::Point(tracker_point_2d.x + crosshair_size, tracker_point_2d.y),
                            cv::Scalar(0, 0, 255), 2);
                    cv::line(image, 
                            cv::Point(tracker_point_2d.x, tracker_point_2d.y - crosshair_size),
                            cv::Point(tracker_point_2d.x, tracker_point_2d.y + crosshair_size),
                            cv::Scalar(0, 0, 255), 2);

                    // Draw velocity vector if available (predict 0.2s ahead for visualization)
                    if(std::abs(track_data_.velocity.x) > 0.01 || 
                       std::abs(track_data_.velocity.y) > 0.01 || 
                       std::abs(track_data_.velocity.z) > 0.01)
                    {
                        // Predict position 0.2 second ahead from compensated position
                        geometry_msgs::Point predicted_pos;
                        predicted_pos.x = compensated_pos.x + track_data_.velocity.x * 0.2;
                        predicted_pos.y = compensated_pos.y + track_data_.velocity.y * 0.2;
                        predicted_pos.z = compensated_pos.z + track_data_.velocity.z * 0.2;

                        cv::Point2d predicted_point_2d;
                        if(projectPoint3Dto2D(predicted_pos, predicted_point_2d))
                        {
                            // Draw arrow showing velocity direction
                            cv::arrowedLine(image, tracker_point_2d, predicted_point_2d, 
                                          cv::Scalar(255, 0, 255), 2, cv::LINE_AA, 0, 0.3);
                        }
                    }

                    // Draw tracking info text near tracker point
                    std::stringstream ss;
                    ss << "TRACKER";
                    cv::putText(image, ss.str(), 
                               cv::Point(tracker_point_2d.x + 20, tracker_point_2d.y - 10),
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
                }
                else
                {
                    cv::putText(image, "Tracker out of bounds", cv::Point(10, panel_y),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
                }
            }
            else
            {
                cv::putText(image, "Projection failed (TF error?)", cv::Point(10, panel_y),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }
        }
        else
        {
            // Draw "NO TRACKING" text
            cv::putText(image, "NO TRACKING", cv::Point(10, panel_y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
    }


    void Processor::drawconfigCB(rm_radar_img_proc::DrawConfig& config, uint32_t level)
    {
        draw_type_ = DrawImage(config.draw_type);
        line_width_ = config.line_width;

        if (show_fps_ != config.show_fps)
        {
            show_fps_ = config.show_fps;
            fps_ema_ = 0.0;
            last_fps_stamp_ = ros::Time();
        }
        else
        {
            show_fps_ = config.show_fps;
        }
    }


    void Processor::draw()
    {
        cv::Mat draw_image;
        sensor_msgs::ImagePtr msg;
        if(is_tele_cam_)
        {
            ROS_INFO_ONCE("[mode] tele camera draw image");
            if (draw_type_ == DrawImage::RAW)
            {
                raw_image_.copyTo(draw_image);
            }
            else if(draw_type_ == DrawImage::BINARY)
            {
                binary_image_.copyTo(draw_image);
            }
            else if (draw_type_ == DrawImage::MORPHOLOGY)
            {
                cv::cvtColor(morpro_image_, draw_image, cv::COLOR_GRAY2BGR);
            }
            else if (draw_type_ == DrawImage::BARS)
            {
                raw_image_.copyTo(draw_image);
                drawBars(draw_image);
            }
            else if (draw_type_ == DrawImage::ARMORS)
            {
                raw_image_.copyTo(draw_image);
                drawArmors(draw_image);
            }
            else if (draw_type_ == DrawImage::ARMORS_VERTEXS)
            {
                raw_image_.copyTo(draw_image);
                drawArmorsVertexes(draw_image);
            }
            else if (draw_type_ == DrawImage::WARP)
            {
                if (!drawWarp())
                {
                    ROS_WARN("cannot draw warp image, because armors size != 1");
                    raw_image_.copyTo(draw_image);
                }
                else if (!warp_image_.empty())
                {
                    cv::imshow("warp_image", warp_image_);
                    cv::waitKey(1);
                    raw_image_.copyTo(draw_image);
                    drawArmors(draw_image);
                }
                else
                {
                    ROS_WARN("warp_image is empty");
                    raw_image_.copyTo(draw_image);
                }
            }
            else if (draw_type_ == DrawImage::TRACKER)
            {
                raw_image_.copyTo(draw_image);
                drawTracker(draw_image);
            }
            else
            {
                raw_image_.copyTo(draw_image);
                drawArmors(draw_image);
                drawBars(draw_image);
            }
        }

        if (show_fps_ && !draw_image.empty())
        {
            const ros::WallTime now_wall = ros::WallTime::now();
            static ros::WallTime last_wall = now_wall;

            const double dt = (now_wall - last_wall).toSec();
            last_wall = now_wall;

            if (dt > 1e-4 && dt < 1.0)
            {
                const double fps_inst = 1.0 / dt;
                const double alpha = 0.1;
                fps_ema_ = (fps_ema_ <= 1e-6) ? fps_inst : (alpha * fps_inst + (1.0 - alpha) * fps_ema_);
            }

            char buf[64];
            std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps_ema_);
            cv::putText(draw_image, buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 255, 255), 2);
        }

        if (!draw_image.empty())
        {
            const bool is_mono = (draw_type_ == DrawImage::BINARY);
            msg = cv_bridge::CvImage(std_msgs::Header(), is_mono ? "mono8" : "bgr8", draw_image).toImageMsg();
        }
        if(msg)
        {
            image_pub_.publish(msg);
        }
    }
    cv::Mat Processor::setElement()
    {
        return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(binary_element_, binary_element_), cv::Point(-1, -1));
    }

    void Processor::hsv2Binary()
    {
        cv::Mat hsv_image;

        cvtColor(this->raw_image_, hsv_image, cv::COLOR_BGR2HSV);
        if (target_is_red_ == 1)
        {
            cv::Mat h_binary_low, h_binary_high;
            inRange(hsv_image, cv::Scalar(red_h_min_low_, red_s_min_, red_v_min_), cv::Scalar(red_h_max_low_, red_s_max_, red_v_max_),
                    h_binary_low);
            inRange(hsv_image, cv::Scalar(red_h_min_high_, red_s_min_, red_v_min_), cv::Scalar(red_h_max_high_, red_s_max_, red_v_max_),
                    h_binary_high);
            bitwise_or(h_binary_low, h_binary_high, binary_image_);
        }
        else
        {
            inRange(hsv_image, cv::Scalar(blue_h_min_, blue_s_min_, blue_v_min_), cv::Scalar(blue_h_max_, blue_s_max_, blue_v_max_),
                    binary_image_);
        }
    }

    void Processor::bgr2Binary()
    {
        std::vector<cv::Mat> channels;

        cv::split(this->raw_image_, channels);
        if (target_is_red_ == 1)
            binary_image_ = channels[2] - channels[0];
        else
            binary_image_ = channels[0] - channels[2];
        threshold(binary_image_, binary_image_, binary_thresh_, 255, cv::THRESH_BINARY);
    }

    void Processor::imageProcess(cv_bridge::CvImagePtr &cv_image)
    {
        cv::Mat element = setElement();
        cv_image->image.copyTo(raw_image_);
        if (gamma_)
            cv::LUT(raw_image_, look_up_table_, raw_image_);
        switch (preprocess_method_)
        {
            case PreProcessMethod::HSV:
            {
            hsv2Binary();
            break;
            }
            case PreProcessMethod::SINGLE_CHANNEL:
            {
            bgr2Binary();
            break;
            }
        }
        if (morph_type_ == 0)
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_ERODE, element);
        else if (morph_type_ == 1)
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_DILATE, element);
        else if (morph_type_ == 2)
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_OPEN, element);
        else if (morph_type_ == 3)
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_CLOSE, element);
        else if (morph_type_ == 4)
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_GRADIENT, element);
        else if (morph_type_ == 5)
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_TOPHAT, element);
        else if (morph_type_ == 6)
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_BLACKHAT, element);
        else if (morph_type_ == 7)
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_HITMISS, element);
        else if (morph_type_ == 8)
            binary_image_.copyTo(morpro_image_);
    }



    bool Processor::isValidBar(const Bar& bar)
    {   
        double angle = static_cast<double>(abs(bar.angle_));
        
        double bar_angle_diff_ = static_cast<double>(abs(abs(angle) - 90.0));
        if (is_bar_debug_)
        {
            ROS_INFO("bar_angle: %lf",  bar.angle_);
            ROS_INFO("bar_angle_diff: %.2f (max_allowed: %.2f) %s", 
                     bar_angle_diff_, max_angle_diff_,
                     bar_angle_diff_ > max_angle_diff_ ? "-> REJECT" : "-> OK");
            ROS_INFO("lw_ratio: %.2f (min: %.2f, max: %.2f) %s", 
                     bar.lw_ratio_, min_lw_ratio_, max_lw_ratio_,
                     (bar.lw_ratio_ < min_lw_ratio_ || bar.lw_ratio_ > max_lw_ratio_) ? "-> REJECT" : "-> OK");
            ROS_INFO("pixel_contained_ratio: %.3f (min: %.3f) %s", 
                     bar.pixel_contained_ratio_, min_pixel_contained_ratio_,
                     bar.pixel_contained_ratio_ < min_pixel_contained_ratio_ ? "-> REJECT" : "-> OK");
            ROS_INFO("/////////////////////");
        }
        if (bar_angle_diff_ > max_angle_diff_)
            return false;
        if (bar.lw_ratio_ < min_lw_ratio_ || bar.lw_ratio_ > max_lw_ratio_)
            return false;
        if (bar.pixel_contained_ratio_ < min_pixel_contained_ratio_)
            return false;
        return true;
    }

    void Processor::findbars()
    {
        bars_.clear();
        contours_.clear();  
        std::vector<std::vector<cv::Point>> contours;
        findContours(morpro_image_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        ROS_INFO_THROTTLE(3, "[findbars] Total contours: %lu, select_bar=%d", contours.size(), select_bar_);
        cv::RotatedRect rect;
        for(size_t i=0; i<contours.size(); i++)
        {
            double area = contourArea(contours[i]);
                if (area < 5) continue; 
                contours_.push_back(contours[i]); 
                rect = cv::minAreaRect(contours[i]);
                ArmorColor bar_color = target_is_red_ ? ArmorColor::RED : ArmorColor::BLUE;
                Bar bar(rect, contours[i], bar_color);
                if (select_bar_ && !isValidBar(bar)) 
                {
                    continue;
                }
                
                bars_.emplace_back(bar);
            }
        ROS_INFO_THROTTLE(3, "[findbars] Contours: %lu -> Valid bars: %lu", contours.size(), bars_.size());
    }


    bool Processor::isValidArmor(Bar& top_bar, Bar& bottom_bar)
    {
        double distance = pow(top_bar.center_point_.x - bottom_bar.center_point_.x, 2) +
                          pow(top_bar.center_point_.y - bottom_bar.center_point_.y, 2);
        distance = sqrt(distance);

        double lens = top_bar.length_len_ + bottom_bar.length_len_;

        double bars_angle = fabs(abs(top_bar.angle_) - abs(bottom_bar.angle_));

        if (is_armor_debug_)
        {
            ROS_INFO("max_bars_ratio: %lf > %lf", max_bars_ratio_,
                    std::max(top_bar.length_len_, bottom_bar.length_len_) / std::min(top_bar.length_len_, bottom_bar.length_len_));
            // ROS_INFO("distance: %lf > %lf > %lf ", bars_length * max_bars_distance_, distance, bars_length * min_bars_distance_);
            ROS_INFO("top_bar: (%f, %f)", top_bar.center_point_.x, top_bar.center_point_.y);
            ROS_INFO("bottom_bar: (%f, %f)", bottom_bar.center_point_.x, bottom_bar.center_point_.y);
            ROS_INFO("max_bars_angle: %lf > %lf", max_bars_angle_, bars_angle);
            ROS_INFO("top_bar_angle: %f", top_bar.angle_);
            ROS_INFO("bottom_bar_angle: %f", bottom_bar.angle_);
            ROS_INFO("max_bars_x_dis: %lf > %lf", max_bars_x_dis_,
                    fabs(top_bar.center_point_.x - bottom_bar.center_point_.x) / (lens * 0.5));
            ROS_INFO("////////////////////////////////////////////");
        }

        if (std::max(top_bar.length_len_, bottom_bar.length_len_) / std::min(top_bar.length_len_, bottom_bar.length_len_) > max_bars_ratio_)
            {
                if(is_armor_debug_)
                {ROS_INFO("failed in bars ratio, %lf > max bar ratio : %lf", std::max(top_bar.length_len_, bottom_bar.length_len_) / std::min(top_bar.length_len_, bottom_bar.length_len_), max_bars_ratio_);}
            return false;
            }

        if (distance > lens * max_bars_distance_ || distance < lens * min_bars_distance_)
            {
            if(is_armor_debug_)
                {ROS_INFO("failed in bars distance, %lf > max bars distance : %lf", distance, lens * max_bars_distance_);}
            return false;
            }

        // if (bars_angle > max_bars_angle_)
        //     {
        //     if(is_armor_debug_)
        //         {ROS_INFO("failed in bars angle, %lf > max bars angle : %lf", bars_angle, max_bars_angle_);}
        //     return false;
        //     }

        if (fabs(top_bar.center_point_.x - bottom_bar.center_point_.x) / (lens * 0.5) > max_bars_x_dis_)
            {
            if(is_armor_debug_)
                {ROS_INFO("failed in bars x dis, %lf > max bars x dis : %lf", fabs(top_bar.center_point_.x - bottom_bar.center_point_.x) / (lens * 0.5), max_bars_x_dis_);}
            return false;
            }

        
        Armor armor_tmp(top_bar, bottom_bar);
        if(armor_tmp.parallel_dist_/ distance > max_bars_ratio_)
        {   if(is_armor_debug_)
                {ROS_INFO("failed in bars parallel dist, %lf > max bars parallel dist : %lf", armor_tmp.parallel_dist_/ distance, max_bars_ratio_);}
            return false;
        }

        return true;
    }

    
    void Processor::solvePose(const Armor& armor, rm_radar_msgs::DroneDetection& target)
    {
        if(intrinsics_.empty() || dist_coeffs_.empty())
        {
            ROS_WARN("[PNP] camera intrinsics or dist coeffs is empty, cannot solve pnp");
            return;
        }

        cv::Mat rvec, tvec;
        std::vector<cv::Point2d> img_points;
        for(size_t i=0; i<4; i++)
        {
            img_points.emplace_back(cv::Point2d(armor.bars_4points_[i].x, armor.bars_4points_[i].y));
        }

        bool success = cv::solvePnP(armor_3d_points_, img_points, intrinsics_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE);
        if(success)
        {
            target.pose.position.x = tvec.at<double>(0);                               
            target.pose.position.y = tvec.at<double>(1);
            target.pose.position.z = tvec.at<double>(2);

            Eigen::Vector3d rvec_eigen;
            cv::cv2eigen(rvec, rvec_eigen);
            double angle = rvec_eigen.norm();
            Eigen::Vector3d axis = rvec_eigen / angle;
            Eigen::Quaterniond q;
            if(angle <1e-6)
            {
                q = Eigen::Quaterniond::Identity();
            }
            else
            {
                q = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
            }
            target.pose.orientation.x = q.x();
            target.pose.orientation.y = q.y();
            target.pose.orientation.z = q.z();
            target.pose.orientation.w = q.w();

        }
        else
        {
            ROS_WARN("[PNP] solve pnp failed");
        }

    }

    double Processor::getArmorScore(const Bar& bar_top, const Bar& bar_bottom)
    {
        double len_avg = (bar_top.length_len_ + bar_bottom.length_len_) / 2.0;
        double x_dif = fabs(bar_top.center_point_.x - bar_bottom.center_point_.x);
        double angle_diff = fabs(abs(bar_top.angle_) - abs(bar_bottom.angle_));
        double len_diff = fabs(bar_top.length_len_ - bar_bottom.length_len_);
        double center_angle = atan2(bar_bottom.center_point_.y - bar_top.center_point_.y, bar_bottom.center_point_.x - bar_top.center_point_.x) * 180.0 / CV_PI;
        double vertical_deviation = fabs(abs(center_angle) - 90.0);

        double score = 0;
        score += (1.0 - x_dif / (len_avg * max_bars_x_dis_)) * 0.4;  // x差距越小越好
        score += (1.0 - angle_diff / max_bars_angle_) * 0.3;  // 角度差距越小越好
        score += (1.0 - len_diff / (len_avg * max_bars_ratio_)) * 0.2;  // 长度差距越小越好
        score += (1.0 - vertical_deviation / max_bars_angle_) * 0.1;  // 垂直偏差越小越好
        score = std::max(0.0, std::min(1.0, score));  // 将分数限制在0到1之间
    
        return score;
    }

    void Processor::findArmor()
    {
        this->armors_.clear();
        this->target_array_.detections.clear();
        findbars();
        if(bars_.size() < 2)
        {
            ROS_WARN_THROTTLE(2, "[findArmor] Only %lu bars found (need >=2). No armor detection possible.", bars_.size());
            return;
        }
        ROS_INFO_THROTTLE(2, "[findArmor] Found %lu bars, searching for armor pairs...", bars_.size());
        std::sort(bars_.begin(), bars_.end(), [](const Bar& a, const Bar& b) {
            return a.center_point_.y < b.center_point_.y;
        });

        // Collect all valid armor pairs with their scores
        std::vector<MatchPairs> match_pairs;
        for (size_t i = 0; i < bars_.size(); i++)
        {
            for (size_t j = i + 1; j < bars_.size(); j++)
            {
                Bar& bar_top = bars_[i];
                Bar& bar_bottom = bars_[j];
                if (isValidArmor(bar_top, bar_bottom))
                {
                    double score = getArmorScore(bar_top, bar_bottom);
                    match_pairs.emplace_back(i, j, score);
                }
            }
        }

        // Sort by score (highest first) and greedily select non-overlapping pairs
        std::sort(match_pairs.begin(), match_pairs.end(), [](const MatchPairs& a, const MatchPairs& b) {
            return a.score > b.score;
        });

        double tmp_score = 0.0;
        std::vector<bool> bar_used(bars_.size(), false);
        for (const auto& pair : match_pairs)
        {
            if (bar_used[pair.index_top] || bar_used[pair.index_bottom])
                continue;
            
            tmp_score = pair.score;
            Bar& bar_top = bars_[pair.index_top];
            Bar& bar_bottom = bars_[pair.index_bottom];
            Armor armor_tmp(bar_top, bar_bottom);
            armor_tmp.confidence_ = tmp_score;
            this->armors_.emplace_back(armor_tmp);
            bar_used[pair.index_top] = true;
            bar_used[pair.index_bottom] = true;
        }

        if (armors_.empty()) {
            ROS_WARN_THROTTLE(2, "[findArmor] %lu bars found but no valid armor pairs matched! "
                              "(check isValidArmor criteria: angle/ratio/size thresholds)", bars_.size());
            return;
        }
        ROS_INFO_THROTTLE(2, "[findArmor] Matched %lu armors from %lu bars", armors_.size(), bars_.size());

        for(auto& armor : armors_)
        {
            // warp up data
            double x_center = (armor.bars_4points_[0].x + armor.bars_4points_[2].x) / 2.0;
            double y_center = (armor.bars_4points_[0].y + armor.bars_4points_[2].y) / 2.0;
            double distance2ImgCenter = sqrt(pow(x_center - image_x_center_, 2) + pow(y_center - image_y_center_, 2));

            rm_radar_msgs::DroneDetection target;
            target.header = target_array_.header;  // 设置 header，PoseSolver 需要用于 TF
            target.id = armor.id_;
            target.confidence = armor.confidence_;
            target.distance_to_image_center = distance2ImgCenter;
            
            // Store pixel coordinates in x_offset/y_offset
            target.x_offset = static_cast<uint32_t>(armor.center_.x);
            target.y_offset = static_cast<uint32_t>(armor.center_.y);

            for(size_t i=0; i<4; i++)
            {
                target.armor_points[i].x = static_cast<int32_t>(armor.bars_4points_[i].x);
                target.armor_points[i].y = static_cast<int32_t>(armor.bars_4points_[i].y);
                // target.armor_points[i].z = 0;
            }
            // PnP求解移至 PoseSolver，这里只发布2D点
            solvePose(armor, target);
            
            // Set centroid to the same 3D position as pose (camera frame)
            target.centroid.x = target.pose.position.x;
            target.centroid.y = target.pose.position.y;
            target.centroid.z = target.pose.position.z;
            
            target_pub_single_.publish(target);
            target_array_.detections.push_back(target);
        }
        target_array_.is_red = target_is_red_;
        target_pub_.publish(target_array_);

    }

    rm_vision::ProcessorInterface::Object Processor::getObj()
    {
        std::lock_guard<std::mutex> guard(this->obj_locker_);
        return this->object_;
    }

    void Processor::putObj()
    {
        std::lock_guard<std::mutex> guard(this->obj_locker_);
        this->object_.points = points_;
        this->object_.prob = probs_;
        this->object_.id = labels_;
    }


    void Processor::paramReconfig()
    {
    }


}  // namespace rm_radar_imgProc