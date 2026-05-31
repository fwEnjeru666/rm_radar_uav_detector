#include <rm_radar_imgProc.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

PLUGINLIB_EXPORT_CLASS(rm_radarplugin::Processor, nodelet::Nodelet);

namespace rm_radarplugin
{
    void Processor::onInit()
    {
        ros::NodeHandle& nh = getMTPrivateNodeHandle();
        static ros::CallbackQueue callback_queue;
        nh.setCallbackQueue(&callback_queue);
        initialize(nh);
        my_thread_ = std::thread([this]() {
            ros::SingleThreadedSpinner spinner;
            spinner.spin(&callback_queue);
        });
    }

    void Processor::initialize(ros::NodeHandle &nh)
    {
        nh_ = ros::NodeHandle(nh, "radar_imgProc");
        ros::NodeHandle armor_nh(nh_, "armor_condition");
        ros::NodeHandle preprocess_nh(nh_, "preprocess_condition");
        ros::NodeHandle draw_nh(nh_, "draw_condition");
        ROS_INFO("radar image process initialized");

        {
            const auto& armor_cfg_default = rm_radar_img_proc::ArmorConfig::__getDefault__();
            is_bar_debug_ = armor_cfg_default.is_bar_debug;
            is_armor_debug_ = armor_cfg_default.is_armor_debug;
            select_bar_ = armor_cfg_default.select_bar;
            max_angle_diff_ = armor_cfg_default.max_angle_diff;
            min_lw_ratio_ = armor_cfg_default.min_lw_ratio;
            max_lw_ratio_ = armor_cfg_default.max_lw_ratio;
            min_pixel_contained_ratio_ = armor_cfg_default.min_pixel_contained_ratio;
            max_bars_ratio_ = armor_cfg_default.max_bars_ratio;
            min_bars_distance_ = armor_cfg_default.min_bars_distance;
            max_bars_distance_ = armor_cfg_default.max_bars_distance;
            max_bars_angle_ = armor_cfg_default.max_bars_angle;
            max_bars_x_dis_ = armor_cfg_default.max_bars_x_dis;
            select_by_last_ = armor_cfg_default.select_by_last;

            const auto& preprocess_cfg_default = rm_radar_img_proc::PreprocessConfig::__getDefault__();
            target_is_red_ = preprocess_cfg_default.target_color;
            preprocess_method_ = preprocess_cfg_default.preprocess_method;
            red_h_min_low_ = preprocess_cfg_default.red_h_min_low;
            red_h_max_low_ = preprocess_cfg_default.red_h_max_low;
            red_h_min_high_ = preprocess_cfg_default.red_h_min_high;
            red_h_max_high_ = preprocess_cfg_default.red_h_max_high;
            red_s_min_ = preprocess_cfg_default.red_s_min;
            red_s_max_ = preprocess_cfg_default.red_s_max;
            red_v_min_ = preprocess_cfg_default.red_v_min;
            red_v_max_ = preprocess_cfg_default.red_v_max;
            blue_h_min_ = preprocess_cfg_default.blue_h_min;
            blue_h_max_ = preprocess_cfg_default.blue_h_max;
            blue_s_min_ = preprocess_cfg_default.blue_s_min;
            blue_s_max_ = preprocess_cfg_default.blue_s_max;
            blue_v_min_ = preprocess_cfg_default.blue_v_min;
            blue_v_max_ = preprocess_cfg_default.blue_v_max;
            binary_thresh_ = preprocess_cfg_default.binary_thresh;
            morph_type_ = preprocess_cfg_default.morph_type;
            binary_element_ = preprocess_cfg_default.binary_element;
            kernel_shape_ = preprocess_cfg_default.kernel_shape;
            kernel_w_ = preprocess_cfg_default.kernel_w;
            kernel_h_ = preprocess_cfg_default.kernel_h;
            kernel_angle_deg_ = preprocess_cfg_default.kernel_angle_deg;
            morph_iterations_ = preprocess_cfg_default.morph_iterations;

            const auto& draw_cfg_default = rm_radar_img_proc::DrawConfig::__getDefault__();
            draw_type_ = static_cast<DrawImage>(draw_cfg_default.draw_type);
            line_width_ = draw_cfg_default.line_width;
            show_fps_ = draw_cfg_default.show_fps;
            show_centroid_only_ = draw_cfg_default.show_centroid_only;
        }

        auto armor_params_init = [this, &armor_nh]() {
            ROS_INFO("reading armor param");
            armor_nh.getParam("is_bar_debug", is_bar_debug_);
            armor_nh.getParam("is_armor_debug", is_armor_debug_);
            armor_nh.getParam("select_bar", select_bar_);
            armor_nh.getParam("max_angle_diff", max_angle_diff_);
            armor_nh.getParam("min_lw_ratio", min_lw_ratio_);
            armor_nh.getParam("max_lw_ratio", max_lw_ratio_);
            armor_nh.getParam("min_pixel_contained_ratio", min_pixel_contained_ratio_);
            armor_nh.getParam("max_bars_ratio", max_bars_ratio_);
            armor_nh.getParam("max_bars_distance", max_bars_distance_);
            armor_nh.getParam("min_bars_distance", min_bars_distance_);
            armor_nh.getParam("max_bars_angle", max_bars_angle_);
            armor_nh.getParam("max_bars_x_dis", max_bars_x_dis_);
            armor_nh.getParam("select_by_last", select_by_last_);

            // static-only params under ~/radar_imgProc/*
            nh_.getParam("top_light_y", top_light_y_);
            nh_.getParam("roi_width", roi_width_);
            nh_.getParam("roi_height", roi_height_);
            nh_.getParam("expand_ratio", expand_ratio_);
            nh_.getParam("image_x_center", image_x_center_);
            nh_.getParam("image_y_center", image_y_center_);
            nh_.getParam("camera_coordinate", camera_coordinate_);
            ROS_INFO("Armor params reading done");
        };
        auto pre_process_params_init = [this, &preprocess_nh]() {
            ROS_INFO("reading pre-process param");
            preprocess_nh.getParam("target_color", target_is_red_);
            preprocess_nh.getParam("preprocess_method", preprocess_method_);

            preprocess_nh.getParam("red_h_min_low", red_h_min_low_);
            preprocess_nh.getParam("red_h_max_low", red_h_max_low_);
            preprocess_nh.getParam("red_h_min_high", red_h_min_high_);
            preprocess_nh.getParam("red_h_max_high", red_h_max_high_);
            preprocess_nh.getParam("red_s_min", red_s_min_);
            preprocess_nh.getParam("red_s_max", red_s_max_);
            preprocess_nh.getParam("red_v_min", red_v_min_);
            preprocess_nh.getParam("red_v_max", red_v_max_);

            preprocess_nh.getParam("blue_h_min", blue_h_min_);
            preprocess_nh.getParam("blue_h_max", blue_h_max_);
            preprocess_nh.getParam("blue_s_min", blue_s_min_);
            preprocess_nh.getParam("blue_s_max", blue_s_max_);
            preprocess_nh.getParam("blue_v_min", blue_v_min_);
            preprocess_nh.getParam("blue_v_max", blue_v_max_);

            preprocess_nh.getParam("binary_thresh", binary_thresh_);
            preprocess_nh.getParam("morph_type", morph_type_);
            preprocess_nh.getParam("binary_element", binary_element_);
            preprocess_nh.getParam("kernel_shape", kernel_shape_);
            preprocess_nh.getParam("kernel_w", kernel_w_);
            preprocess_nh.getParam("kernel_h", kernel_h_);
            preprocess_nh.getParam("kernel_angle_deg", kernel_angle_deg_);
            preprocess_nh.getParam("morph_iterations", morph_iterations_);

            binary_element_ = sanitizeKernelSize(binary_element_);
            if (kernel_w_ > 0) kernel_w_ = sanitizeKernelSize(kernel_w_);
            if (kernel_h_ > 0) kernel_h_ = sanitizeKernelSize(kernel_h_);
            kernel_angle_deg_ = ((kernel_angle_deg_ % 180) + 180) % 180;
            morph_iterations_ = std::max(1, morph_iterations_);
            ROS_INFO("pre-processing param reading done");
        };
        auto draw_params_init = [this, &draw_nh]() {
            int draw_type_param = static_cast<int>(draw_type_);
            draw_nh.getParam("draw_type", draw_type_param);
            draw_type_ = static_cast<DrawImage>(draw_type_param);
            draw_nh.getParam("line_width", line_width_);
            draw_nh.getParam("show_fps", show_fps_);
            draw_nh.getParam("show_centroid_only", show_centroid_only_);
        };

        armor_params_init();
        pre_process_params_init();
        draw_params_init();

        armor_cfg_srv_ = std::make_unique<dynamic_reconfigure::Server<rm_radar_img_proc::ArmorConfig>>(ros::NodeHandle(nh_, "armor_condition"));
        armor_cfg_cb_ = boost::bind(&Processor::armorconfigCB, this, _1, _2);
        armor_cfg_srv_->setCallback(armor_cfg_cb_);
        rm_radar_img_proc::ArmorConfig armor_cfg_init;
        armor_cfg_init.is_bar_debug = is_bar_debug_;
        armor_cfg_init.is_armor_debug = is_armor_debug_;
        armor_cfg_init.select_bar = select_bar_;
        armor_cfg_init.max_angle_diff = max_angle_diff_;
        armor_cfg_init.min_lw_ratio = min_lw_ratio_;
        armor_cfg_init.max_lw_ratio = max_lw_ratio_;
        armor_cfg_init.min_pixel_contained_ratio = min_pixel_contained_ratio_;
        armor_cfg_init.max_bars_ratio = max_bars_ratio_;
        armor_cfg_init.min_bars_distance = min_bars_distance_;
        armor_cfg_init.max_bars_distance = max_bars_distance_;
        armor_cfg_init.max_bars_angle = max_bars_angle_;
        armor_cfg_init.max_bars_x_dis = max_bars_x_dis_;
        armor_cfg_init.select_by_last = select_by_last_;
        armor_cfg_srv_->updateConfig(armor_cfg_init);

        preprocess_cfg_srv_ = std::make_unique<dynamic_reconfigure::Server<rm_radar_img_proc::PreprocessConfig>>(ros::NodeHandle(nh_, "preprocess_condition"));
        preprocess_cfg_cb_ = boost::bind(&Processor::preProcessconfigCB, this, _1, _2);
        preprocess_cfg_srv_->setCallback(preprocess_cfg_cb_);
        rm_radar_img_proc::PreprocessConfig preprocess_cfg_init;
        preprocess_cfg_init.target_color = target_is_red_;
        preprocess_cfg_init.preprocess_method = preprocess_method_;
        preprocess_cfg_init.red_h_min_low = red_h_min_low_;
        preprocess_cfg_init.red_h_max_low = red_h_max_low_;
        preprocess_cfg_init.red_h_min_high = red_h_min_high_;
        preprocess_cfg_init.red_h_max_high = red_h_max_high_;
        preprocess_cfg_init.red_s_min = red_s_min_;
        preprocess_cfg_init.red_s_max = red_s_max_;
        preprocess_cfg_init.red_v_min = red_v_min_;
        preprocess_cfg_init.red_v_max = red_v_max_;
        preprocess_cfg_init.blue_h_min = blue_h_min_;
        preprocess_cfg_init.blue_h_max = blue_h_max_;
        preprocess_cfg_init.blue_s_min = blue_s_min_;
        preprocess_cfg_init.blue_s_max = blue_s_max_;
        preprocess_cfg_init.blue_v_min = blue_v_min_;
        preprocess_cfg_init.blue_v_max = blue_v_max_;
        preprocess_cfg_init.binary_thresh = binary_thresh_;
        preprocess_cfg_init.morph_type = morph_type_;
        preprocess_cfg_init.binary_element = binary_element_;
        preprocess_cfg_init.kernel_shape = kernel_shape_;
        preprocess_cfg_init.kernel_w = kernel_w_;
        preprocess_cfg_init.kernel_h = kernel_h_;
        preprocess_cfg_init.kernel_angle_deg = kernel_angle_deg_;
        preprocess_cfg_init.morph_iterations = morph_iterations_;
        preprocess_cfg_srv_->updateConfig(preprocess_cfg_init);

        draw_cfg_srv_ = std::make_unique<dynamic_reconfigure::Server<rm_radar_img_proc::DrawConfig>>(ros::NodeHandle(nh_, "draw_condition"));
        draw_cfg_cb_ = boost::bind(&Processor::drawconfigCB, this, _1, _2);
        draw_cfg_srv_->setCallback(draw_cfg_cb_);
        rm_radar_img_proc::DrawConfig draw_cfg_init;
        draw_cfg_init.draw_type = static_cast<int>(draw_type_);
        draw_cfg_init.line_width = line_width_;
        draw_cfg_init.show_fps = show_fps_;
        draw_cfg_init.show_centroid_only = show_centroid_only_;
        draw_cfg_srv_->updateConfig(draw_cfg_init);

        it_ = std::make_shared<image_transport::ImageTransport>(nh_);
        image_pub_ = it_->advertise("debug_image", 1);

        //cam_sub
        cam_sub_ = it_->subscribeCamera("/hk_camera/image_raw", 1, &Processor::cam_callback, this);

        //tracker_sub
        track_sub_ = nh.subscribe("/tracker/track_data", 1, &Processor::trackerCB, this);

        tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(ros::Duration(10));
        tf_listener_ = std::unique_ptr<tf2_ros::TransformListener>(new tf2_ros::TransformListener(*tf2_buffer_));
        target_pub_ = nh.advertise<decltype(target_array_)>("/processor/result_msg", 10);
        target_pub_single_ = nh.advertise<rm_radar_msgs::DroneDetection>("/processor/single_result_msg", 10);

    }

    //dynamic reconfigure callback
    void Processor::armorconfigCB(rm_radar_img_proc::ArmorConfig& config, uint32_t level)
    {
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

        if (!armor_dynamic_reconfig_initialized_)
        {
            armor_dynamic_reconfig_initialized_ = true;
        }

        ROS_INFO("Armor dynamic reconfigure callback done");
    }

    void Processor::preProcessconfigCB(rm_radar_img_proc::PreprocessConfig& config, uint32_t level)
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
        kernel_shape_ = config.kernel_shape;
        kernel_w_ = config.kernel_w;
        kernel_h_ = config.kernel_h;
        kernel_angle_deg_ = config.kernel_angle_deg;
        morph_iterations_ = config.morph_iterations;

        if (!pre_process_dynamic_reconfig_initialized_)
        {
            pre_process_dynamic_reconfig_initialized_ = true;
        }

        binary_element_ = sanitizeKernelSize(binary_element_);
        if (kernel_w_ > 0) kernel_w_ = sanitizeKernelSize(kernel_w_);
        if (kernel_h_ > 0) kernel_h_ = sanitizeKernelSize(kernel_h_);
        kernel_angle_deg_ = ((kernel_angle_deg_ % 180) + 180) % 180;
        morph_iterations_ = std::max(1, morph_iterations_);
        morph_kernel_dirty_ = true;
    }


    void Processor::drawBars(cv::Mat& image, std::vector<Bar>& bars)
    {
        cv::Scalar line_color = cv::Scalar(0, 255, 0);
        for (const auto& bar : bars)
            for (int j = 0; j < 4; j++)
            cv::line(image, bar.points_[j], bar.points_[(j + 1) % 4], line_color, line_width_);

        if (is_bar_debug_)
        {
            for (const auto& bar : bars)
            {
                cv::Point origin(cvRound(bar.center_point_.x) + 10, cvRound(bar.center_point_.y) - 10);
                origin.x = std::max(5, std::min(origin.x, std::max(5, image.cols - 700)));
                origin.y = std::max(20, std::min(origin.y, std::max(20, image.rows - 10)));
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2)
                    << "BAR ang=" << bar.angle_
                    << " lw=" << bar.lw_ratio_
                    << " pix=" << bar.pixel_contained_ratio_;
                cv::putText(image,
                            oss.str(),
                            origin,
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.55,
                            cv::Scalar(0, 255, 255),
                            2);
            }
        }
    }

    void Processor::drawArmors(cv::Mat& image, std::vector<Armor>& armors)
    {
        cv::Scalar line_color = cv::Scalar(0, 255, 0);
        for (const auto& armor : armors)
        {
            for (int j = 0; j < 4; j++)
                cv::line(image, armor.bars_4points_[j], armor.bars_4points_[(j + 1) % 4], line_color, line_width_);

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

        if (is_armor_debug_)
        {
            for (const auto& armor : armors)
            {
                double bar_ratio = 0.0;
                double bars_dist = 0.0;
                double bars_angle = 0.0;
                double bars_x_dis = 0.0;
                if (armor.bar_up_ != nullptr && armor.bar_bottom_ != nullptr)
                {
                    const double top_len = armor.bar_up_->length_len_;
                    const double bottom_len = armor.bar_bottom_->length_len_;
                    const double min_len = std::min(top_len, bottom_len);
                    const double max_len = std::max(top_len, bottom_len);
                    bar_ratio = (min_len > 1e-6) ? (max_len / min_len) : 0.0;

                    const double dx = static_cast<double>(armor.bar_up_->center_point_.x - armor.bar_bottom_->center_point_.x);
                    const double dy = static_cast<double>(armor.bar_up_->center_point_.y - armor.bar_bottom_->center_point_.y);
                    bars_dist = std::sqrt(dx * dx + dy * dy);

                    bars_angle = std::fabs(static_cast<double>(armor.bar_up_->angle_ - armor.bar_bottom_->angle_));
                    if (bars_angle > 90.0) bars_angle = 180.0 - bars_angle;

                    const double lens = top_len + bottom_len;
                    bars_x_dis = (lens > 1e-6)
                                 ? (std::fabs(dx) / (lens * 0.5))
                                 : 0.0;
                }

                cv::Point origin(cvRound(armor.center_.x) + 12, cvRound(armor.center_.y) - 10);
                origin.x = std::max(5, std::min(origin.x, std::max(5, image.cols - 900)));
                origin.y = std::max(20, std::min(origin.y, std::max(20, image.rows - 10)));
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2)
                    << "ARMOR ratio=" << bar_ratio
                    << " dist=" << bars_dist
                    << " ang=" << bars_angle
                    << " xdis=" << bars_x_dis
                    << " conf=" << armor.confidence_;
                cv::putText(image,
                            oss.str(),
                            origin,
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.55,
                            cv::Scalar(255, 255, 0),
                            2);
            }
        }
    }
    void Processor::drawArmorsVertexes(cv::Mat& image, std::vector<Armor>& armors)
    {
    cv::Scalar vertex_color = cv::Scalar(0, 255, 0);
    int cross_size = 15;
    cv::line(image, cv::Point(image_center_.x - cross_size, image_center_.y), 
                    cv::Point(image_center_.x + cross_size, image_center_.y), cv::Scalar(0, 0, 255), line_width_);
    cv::line(image, cv::Point(image_center_.x, image_center_.y - cross_size), 
                    cv::Point(image_center_.x, image_center_.y + cross_size), cv::Scalar(0, 0, 255), line_width_);
    for (const auto& armor : armors)
    {
        if(!show_centroid_only_)
        {
            for (int i = 0; i < 4; i++)
            {
            circle(image, armor.bars_4points_[i], 3, vertex_color, line_width_);
            putText(image, std::to_string(i), armor.bars_4points_[i], cv::FONT_HERSHEY_COMPLEX, 1.5, vertex_color);
            }
            line(image, armor.bars_4points_[0], armor.bars_4points_[1], cv::Scalar(0, 255, 0), line_width_);
            line(image, armor.bars_4points_[2], armor.bars_4points_[3], cv::Scalar(0, 255, 0), line_width_);
            putText(image, std::to_string(5), armor.center_, cv::FONT_HERSHEY_COMPLEX, 1.5, vertex_color);
        }
        circle(image, armor.center_, 3, vertex_color, line_width_);
        line(image, armor.center_, image_center_, cv::Scalar(255, 0, 0), line_width_);
    }
    }


    void Processor::trackerCB(const rm_msgs::TrackData::ConstPtr& msg)
    {
        // Store the latest track data
        track_data_ = *msg;
        ROS_INFO_ONCE("[Tracker] Received track data");
    }

    void Processor::drawTracker(cv::Mat& image, cv::Point3f& tracked_position)
    {
        if (intrinsics_.empty() || dist_coeffs_.empty())
        {
            return;
        }

        std::vector<cv::Point3f> object_points = { tracked_position };
        std::vector<cv::Point2f> image_points;
        cv::projectPoints(object_points, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F), intrinsics_, dist_coeffs_, image_points);
        
        if (!image_points.empty())
        {
            cv::Point2f img_point = image_points[0];
            cv::circle(image, img_point, 8, cv::Scalar(0, 0, 255), -1);

        }

    }


    void Processor::drawconfigCB(rm_radar_img_proc::DrawConfig& config, uint32_t level)
    {
        const int draw_type_clamped = std::max(static_cast<int>(DrawImage::DISABLE),
                                               std::min(config.draw_type, static_cast<int>(DrawImage::TRACKER)));
        draw_type_ = static_cast<DrawImage>(draw_type_clamped);
        line_width_ = config.line_width;
        show_centroid_only_ = config.show_centroid_only;

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
        if (draw_type_ == DrawImage::DISABLE)
        {
            return;
        }

        cv::Mat draw_image;
        sensor_msgs::ImagePtr msg;
        switch (draw_type_)
            {
                case DrawImage::RAW:
                    raw_image_.copyTo(draw_image);
                    break;
                case DrawImage::BINARY:
                    binary_image_.copyTo(draw_image);
                    break;
                case DrawImage::MORPHOLOGY:
                    cv::cvtColor(morpro_image_, draw_image, cv::COLOR_GRAY2BGR);
                    break;
                case DrawImage::BARS:
                    raw_image_.copyTo(draw_image);
                    drawBars(draw_image, bars_);
                    break;
                case DrawImage::ARMORS:
                    raw_image_.copyTo(draw_image);
                    drawArmors(draw_image, armors_);
                    break;
                case DrawImage::ARMORS_VERTEXS:
                    raw_image_.copyTo(draw_image);
                    drawArmorsVertexes(draw_image, armors_);
                    break;
                case DrawImage::BARS_ARMORS:
                    raw_image_.copyTo(draw_image);
                    drawArmors(draw_image, armors_);
                    drawBars(draw_image, bars_);
                    break;
                case DrawImage::TRACKER:
                    if (track_data_.tracking)
                    {
                        raw_image_.copyTo(draw_image);
                        cv::Point3f pos(track_data_.position.x, track_data_.position.y, track_data_.position.z);
                        drawTracker(draw_image, pos);
                        drawArmorsVertexes(draw_image, armors_);
                    }
                    else
                    {
                        raw_image_.copyTo(draw_image);
                        cv::putText(draw_image, "No target tracked", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                        drawArmorsVertexes(draw_image, armors_);
                    }
                    break;
                default:
                    raw_image_.copyTo(draw_image);
                    drawArmors(draw_image, armors_);
                    drawBars(draw_image, bars_);
                    break;
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
    int Processor::sanitizeKernelSize(int value)
    {
        const int clamped = std::max(1, value);
        return (clamped % 2 == 0) ? (clamped + 1) : clamped;
    }

    cv::Mat Processor::buildMorphKernel()
    {
        int w = kernel_w_ > 0 ? kernel_w_ : binary_element_;
        int h = kernel_h_ > 0 ? kernel_h_ : binary_element_;
        w = sanitizeKernelSize(w);
        h = sanitizeKernelSize(h);

        const int shape = std::max(static_cast<int>(ELLIPSE), std::min(kernel_shape_, static_cast<int>(LINE)));
        const int angle = ((kernel_angle_deg_ % 180) + 180) % 180;

        const bool need_rebuild = morph_kernel_dirty_ ||
                                  morph_kernel_cache_.empty() ||
                                  shape != last_kernel_shape_ ||
                                  w != last_kernel_w_ ||
                                  h != last_kernel_h_ ||
                                  angle != last_kernel_angle_deg_;
        if (!need_rebuild)
        {
            return morph_kernel_cache_;
        }

        switch (shape)
        {
            case RECT:
                morph_kernel_cache_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(w, h), cv::Point(-1, -1));
                break;
            case LINE:
            {
                cv::Mat kernel = cv::Mat::zeros(h, w, CV_8UC1);
                const cv::Point center((w - 1) / 2, (h - 1) / 2);
                const double rad = angle * CV_PI / 180.0;
                const cv::Point2d dir(std::cos(rad), std::sin(rad));
                const double half_len = static_cast<double>(std::max(w, h));
                cv::Point p1(cvRound(center.x - dir.x * half_len), cvRound(center.y - dir.y * half_len));
                cv::Point p2(cvRound(center.x + dir.x * half_len), cvRound(center.y + dir.y * half_len));
                cv::clipLine(cv::Size(w, h), p1, p2);
                cv::line(kernel, p1, p2, cv::Scalar(255), 1, cv::LINE_8);
                morph_kernel_cache_ = kernel;
                break;
            }
            case ELLIPSE:
            default:
                morph_kernel_cache_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(w, h), cv::Point(-1, -1));
                break;
        }

        last_kernel_shape_ = shape;
        last_kernel_w_ = w;
        last_kernel_h_ = h;
        last_kernel_angle_deg_ = angle;
        morph_kernel_dirty_ = false;
        return morph_kernel_cache_;
    }

    cv::Mat Processor::setElement()
    {
        return buildMorphKernel();
    }

    void Processor::hsv2Binary()
    {
        cv::cvtColor(this->raw_image_, hsv_image_, cv::COLOR_BGR2HSV);
        if (target_is_red_ == 1)
        {
            cv::UMat h_binary_low, h_binary_high;
            inRange(hsv_image_, cv::Scalar(red_h_min_low_, red_s_min_, red_v_min_), cv::Scalar(red_h_max_low_, red_s_max_, red_v_max_),
                    h_binary_low);
            inRange(hsv_image_, cv::Scalar(red_h_min_high_, red_s_min_, red_v_min_), cv::Scalar(red_h_max_high_, red_s_max_, red_v_max_),
                    h_binary_high);
            bitwise_or(h_binary_low, h_binary_high, binary_image_);
        }
        else
        {
            inRange(hsv_image_, cv::Scalar(blue_h_min_, blue_s_min_, blue_v_min_), cv::Scalar(blue_h_max_, blue_s_max_, blue_v_max_),
                    binary_image_);
        }
    }

    void Processor::bgr2Binary()
    {
        cv::extractChannel(raw_image_, blue_channel_, 0);
        cv::extractChannel(raw_image_, green_channel_, 1);
        cv::extractChannel(raw_image_, red_channel_, 2);

        if (target_is_red_ == 1) {
            // 红 - 绿
            cv::subtract(red_channel_, green_channel_, binary_image_);
        } else {
            // 蓝 - 绿
            cv::subtract(blue_channel_, green_channel_, binary_image_);
        }   

        cv::threshold(binary_image_, binary_image_, binary_thresh_, 255, cv::THRESH_BINARY);
    }

    void Processor::imageProcess(cv_bridge::CvImagePtr &cv_image)
    {
        cv_image->image.copyTo(raw_image_);
        const cv::Mat element = buildMorphKernel();

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
        
        switch (morph_type_)
        {
            case 0:
                morphologyEx(binary_image_, morpro_image_, cv::MORPH_ERODE, element, cv::Point(-1, -1), morph_iterations_);
                break;
            case 1:
                morphologyEx(binary_image_, morpro_image_, cv::MORPH_DILATE, element, cv::Point(-1, -1), morph_iterations_);
                break;
            case 2:
                morphologyEx(binary_image_, morpro_image_, cv::MORPH_OPEN, element, cv::Point(-1, -1), morph_iterations_);
                break;
            case 3:
                morphologyEx(binary_image_, morpro_image_, cv::MORPH_CLOSE, element, cv::Point(-1, -1), morph_iterations_);
                break;
            case 4:
                morphologyEx(binary_image_, morpro_image_, cv::MORPH_GRADIENT, element, cv::Point(-1, -1), morph_iterations_);
                break;
            case 5:
                morphologyEx(binary_image_, morpro_image_, cv::MORPH_TOPHAT, element, cv::Point(-1, -1), morph_iterations_);
                break;
            case 6:
                morphologyEx(binary_image_, morpro_image_, cv::MORPH_BLACKHAT, element, cv::Point(-1, -1), morph_iterations_);
                break;
            case 7:
                morphologyEx(binary_image_, morpro_image_, cv::MORPH_HITMISS, element, cv::Point(-1, -1), morph_iterations_);
                break;
            case 8:
                binary_image_.copyTo(morpro_image_);
                break;
            default:
                break;
        }
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
        if(is_bar_debug_)
        {
            ROS_INFO_THROTTLE(3, "[findbars] Total contours: %lu, select_bar=%d", contours.size(), select_bar_);
        }
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

        if(is_bar_debug_)
        {
            ROS_INFO_THROTTLE(3, "[findbars] Contours: %lu -> Valid bars: %lu", contours.size(), bars_.size());
        }
    }


    bool Processor::isValidArmor(Bar& top_bar, Bar& bottom_bar)
    {
        double distance = cv::norm(top_bar.center_point_ - bottom_bar.center_point_);
        double lens = top_bar.length_len_ + bottom_bar.length_len_;
        double bars_angle = std::fabs(std::fabs(top_bar.angle_) - std::fabs(bottom_bar.angle_));

        if (is_armor_debug_)
        {
            ROS_INFO("max_bars_ratio: %lf > %lf", max_bars_ratio_,
                     std::max(top_bar.length_len_, bottom_bar.length_len_) / std::min(top_bar.length_len_, bottom_bar.length_len_));
            ROS_INFO("min_bars_distance check: distance=%lf, min_limit=%lf",
                     distance, lens * min_bars_distance_);
            ROS_INFO("max_bars_distance check: distance=%lf, max_limit=%lf",
                     distance, lens * max_bars_distance_);
            ROS_INFO("top_bar: (%f, %f)", top_bar.center_point_.x, top_bar.center_point_.y);
            ROS_INFO("bottom_bar: (%f, %f)", bottom_bar.center_point_.x, bottom_bar.center_point_.y);
            ROS_INFO("max_bars_angle: %lf > %lf", max_bars_angle_, bars_angle);
            ROS_INFO("top_bar_angle: %f", top_bar.angle_);
            ROS_INFO("bottom_bar_angle: %f", bottom_bar.angle_);
            ROS_INFO("max_bars_x_dis: %lf > %lf", max_bars_x_dis_,
                     std::fabs(top_bar.center_point_.x - bottom_bar.center_point_.x) / (lens * 0.5));
            ROS_INFO("////////////////////////////////////////////");
        }

        if (std::max(top_bar.length_len_, bottom_bar.length_len_) / std::min(top_bar.length_len_, bottom_bar.length_len_) > max_bars_ratio_)
        {
            if (is_armor_debug_)
                ROS_INFO("failed in bars ratio, %lf > max bar ratio : %lf", 
                         std::max(top_bar.length_len_, bottom_bar.length_len_) / std::min(top_bar.length_len_, bottom_bar.length_len_), max_bars_ratio_);
            return false;
        }

        if (distance > lens * max_bars_distance_ || distance < lens * min_bars_distance_)
        {
            if (is_armor_debug_)
                ROS_INFO("failed in bars distance, %lf > max bars distance : %lf", distance, lens * max_bars_distance_);
            return false;
        }

        if (std::fabs(top_bar.center_point_.x - bottom_bar.center_point_.x) / (lens * 0.5) > max_bars_x_dis_)
        {
            if (is_armor_debug_)
                ROS_INFO("failed in bars x dis, %lf > max bars x dis : %lf", 
                         std::fabs(top_bar.center_point_.x - bottom_bar.center_point_.x) / (lens * 0.5), max_bars_x_dis_);
            return false;
        }
        
        Armor armor_tmp(top_bar, bottom_bar);
        if (armor_tmp.parallel_dist_/ distance > max_bars_ratio_)
        {   
            if (is_armor_debug_)
                ROS_INFO("failed in bars parallel dist, %lf > max bars parallel dist : %lf", armor_tmp.parallel_dist_/ distance, max_bars_ratio_);
            return false;
        }

        return true;
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
        score += (1.0 - x_dif / (len_avg * max_bars_x_dis_)) * 0.4; 
        score += (1.0 - angle_diff / max_bars_angle_) * 0.3; 
        score += (1.0 - len_diff / (len_avg * max_bars_ratio_)) * 0.2;  
        score += (1.0 - vertical_deviation / max_bars_angle_) * 0.1;
    
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
        
        if(is_armor_debug_)
        {
            ROS_INFO_THROTTLE(2, "[findArmor] Found %lu bars, searching for armor pairs...", bars_.size());
        }
        std::sort(bars_.begin(), bars_.end(), [](const Bar& a, const Bar& b) {
            return a.center_point_.y < b.center_point_.y;
        });

        std::vector<MatchPairs> match_pairs;
        match_pairs.reserve(bars_.size() * (bars_.size() - 1) / 2); 

        double max_bar_length = 0.0;
        for (const auto& bar : bars_)
        {
            if (bar.length_len_ > max_bar_length)
            {
                max_bar_length = bar.length_len_;
            }
        }

        for (size_t i = 0; i < bars_.size(); i++)
        {
            Bar& bar_top = bars_[i];
            const double max_pair_dy = (bar_top.length_len_ + max_bar_length) * max_bars_distance_;
            for (size_t j = i + 1; j < bars_.size(); j++)
            {
                Bar& bar_bottom = bars_[j];
                const double dy = bar_bottom.center_point_.y - bar_top.center_point_.y;
                if (dy > max_pair_dy)
                {
                    break;
                }
                if (isValidArmor(bar_top, bar_bottom))
                {
                    double score = getArmorScore(bar_top, bar_bottom);
                    match_pairs.emplace_back(i, j, score);
                }
            }
        }

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
            ROS_WARN_THROTTLE(2, "[findArmor] %lu bars found but no valid armor pairs matched! ", bars_.size());
            return;
        }

        const auto best_it = std::max_element(
            armors_.begin(), armors_.end(),
            [](const Armor& a, const Armor& b) {
                return a.confidence_ < b.confidence_;
            });
        const size_t best_idx = static_cast<size_t>(std::distance(armors_.begin(), best_it));

        if(raw_image_.channels() == 3)
            cv::cvtColor(raw_image_, gray_image_, cv::COLOR_BGR2GRAY);
        else
            raw_image_.copyTo(gray_image_);

        const cv::Size win_size(5, 5); 
        const cv::Size zero_zone(-1, -1);
        const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

        {
            auto& armor = armors_[best_idx];
            std::vector<cv::Point2f> corners;
            corners.reserve(4);
            for (int i = 0; i < 4; ++i) {
                corners.emplace_back(cv::Point2f(armor.bars_4points_[i].x, armor.bars_4points_[i].y));
            }

            cv::cornerSubPix(gray_image_, corners, win_size, zero_zone, criteria);
            for (int i = 0; i < 4; ++i) {
                armor.bars_4points_[i] = cv::Point2d(corners[i].x, corners[i].y);
            }
        }

        if(is_armor_debug_)
        {
            ROS_INFO_THROTTLE(2, "[findArmor] Sub-pixel optimized best armor idx=%lu score=%.3f among %lu armors",
                              best_idx, armors_[best_idx].confidence_, armors_.size());
        }

        target_array_.detections.reserve(armors_.size());

        rm_radar_msgs::DroneDetection best_target;
        bool has_best_target = false;

        for (size_t armor_idx = 0; armor_idx < armors_.size(); ++armor_idx)
        {
            auto& armor = armors_[armor_idx];
            double distance2ImgCenter = sqrt((armor.center_.x - image_center_.x) * (armor.center_.x - image_center_.x) + (armor.center_.y - image_center_.y) * (armor.center_.y - image_center_.y));
            const double raw_error_x = image_center_.x - armor.center_.x;
            const double raw_error_y = image_center_.y - armor.center_.y;

            rm_radar_msgs::DroneDetection target;
            target.header = target_array_.header; 
            target.is_cam_msg = true;
            target.is_lidar_msg = false;
            target.confidence = armor.confidence_;
            target.distance_to_image_center = distance2ImgCenter;

            for(size_t i=0; i<4; i++)
            {
                target.armor_points[i].x = armor.bars_4points_[i].x;
                target.armor_points[i].y = armor.bars_4points_[i].y;
            }
            
            target.target_centroid_x = static_cast<uint32_t>(armor.center_.x);
            target.target_centroid_y = static_cast<uint32_t>(armor.center_.y);
            target.error_x = raw_error_x;
            target.error_y = raw_error_y;
            target.error_angle_yaw   = std::atan(target.error_x / fx_);
            target.error_angle_pitch = std::atan(target.error_y / fy_);
            
            target_array_.detections.push_back(target);

            if (!has_best_target || target.confidence > best_target.confidence)
            {
                best_target = target;
                has_best_target = true;
            }
        }

        if (has_best_target)
        {
            target_pub_single_.publish(best_target);
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
    }


    void Processor::paramReconfig()
    {
    }


}  // namespace rm_radar_imgProc
