#include <rm_radar_imgProc.h>
#include <pluginlib/class_list_macros.h>
#include <algorithm>
#include <opencv2/calib3d.hpp>

PLUGINLIB_EXPORT_CLASS(rm_radarplugin::Processor, nodelet::Nodelet);

namespace rm_radarplugin
{
    void Processor::onInit()
    {
        ros::NodeHandle& nh = getMTPrivateNodeHandle();
        nh.setCallbackQueue(&callback_queue_);
        initialize(nh);
        my_thread_ = std::thread([this]() {
            ros::SingleThreadedSpinner spinner;
            spinner.spin(&callback_queue_);
        });
        worker_thread_ = std::thread(&Processor::processingLoop, this);
    }

    void Processor::initialize(ros::NodeHandle &nh)
    {
        nh_ = ros::NodeHandle(nh, "radar_imgProc");
        nh_.param("detect_method", detect_options_.detect_method, static_cast<int>(DetectMethod::TRADITIONAL));
        nh_.param<std::string>("model_config_path", detect_options_.model_config_path, "");
        nh_.param("process_debug", detect_options_.process_debug, false);
        nh_.param("enable_target_filter", detect_options_.enable_target_filter, true);
        detect_options_.detect_method = std::max(static_cast<int>(DetectMethod::TRADITIONAL),
                                                 std::min(detect_options_.detect_method, static_cast<int>(DetectMethod::YOLO)));
        process_debug_ = detect_options_.process_debug;
        ROS_INFO("radar image process initialized");

        setDynamicReconfig();
        detect_options_.select_bar = select_bar_;
        detect_options_.target_is_red = preprocess_options_.target_is_red;
        detect_options_.object_options = object_options_;
        preprocess_core_ = std::make_unique<PreprocessCore>(preprocess_options_);
        preprocess_options_ = preprocess_core_->options();
        detect_options_.target_is_red = preprocess_options_.target_is_red;
        detect_core_ = std::make_unique<DetectCore>(detect_options_);
        visualizer_ = std::make_unique<tools::Visualizer>(draw_options_);

        it_ = std::make_shared<image_transport::ImageTransport>(nh_);
        image_pub_ = it_->advertise("debug_image", 1);

        //cam_sub
        cam_sub_ = it_->subscribeCamera("/hk_camera/image_raw", 1, &Processor::cam_callback, this);

        target_pub_single_ = nh.advertise<rm_radar_msgs::DroneDetection>("/processor/single_result_msg", 10);
    }

    void Processor::setDynamicReconfig()
    {
        ros::NodeHandle armor_nh(nh_, "armor_condition");
        ros::NodeHandle preprocess_nh(nh_, "preprocess_condition");
        ros::NodeHandle draw_nh(nh_, "draw_condition");

        {
            const auto& armor_cfg_default = rm_radar_img_proc::ArmorConfig::__getDefault__();
            process_debug_ = armor_cfg_default.process_debug;
            detect_options_.process_debug = process_debug_;
            object_options_.bar_enabled = armor_cfg_default.is_bar_debug;
            object_options_.armor_enabled = armor_cfg_default.is_armor_debug;
            select_bar_ = armor_cfg_default.select_bar;
            detect_options_.enable_target_filter = armor_cfg_default.enable_target_filter;
            detect_options_.save_yolo_samples = armor_cfg_default.save_yolo_samples;
            object_options_.max_angle_diff = armor_cfg_default.max_angle_diff;
            object_options_.min_lw_ratio = armor_cfg_default.min_lw_ratio;
            object_options_.max_lw_ratio = armor_cfg_default.max_lw_ratio;
            object_options_.min_pixel_contained_ratio = armor_cfg_default.min_pixel_contained_ratio;
            object_options_.max_bars_ratio = armor_cfg_default.max_bars_ratio;
            object_options_.min_bars_distance = armor_cfg_default.min_bars_distance;
            object_options_.max_bars_distance = armor_cfg_default.max_bars_distance;
            object_options_.max_bars_angle = armor_cfg_default.max_bars_angle;
            object_options_.max_bars_x_dis = armor_cfg_default.max_bars_x_dis;
            detect_options_.refine_max_brightness = armor_cfg_default.refine_max_brightness;
            detect_options_.refine_roi_scale = armor_cfg_default.refine_roi_scale;
            detect_options_.refine_search_start = armor_cfg_default.refine_search_start;
            detect_options_.refine_search_end = armor_cfg_default.refine_search_end;

            const auto& preprocess_cfg_default = rm_radar_img_proc::PreprocessConfig::__getDefault__();
            preprocess_options_.target_is_red = preprocess_cfg_default.target_color;
            detect_options_.target_is_red = preprocess_options_.target_is_red;
            preprocess_options_.preprocess_method = preprocess_cfg_default.preprocess_method;
            preprocess_options_.red_h_min_low = preprocess_cfg_default.red_h_min_low;
            preprocess_options_.red_h_max_low = preprocess_cfg_default.red_h_max_low;
            preprocess_options_.red_h_min_high = preprocess_cfg_default.red_h_min_high;
            preprocess_options_.red_h_max_high = preprocess_cfg_default.red_h_max_high;
            preprocess_options_.red_s_min = preprocess_cfg_default.red_s_min;
            preprocess_options_.red_s_max = preprocess_cfg_default.red_s_max;
            preprocess_options_.red_v_min = preprocess_cfg_default.red_v_min;
            preprocess_options_.red_v_max = preprocess_cfg_default.red_v_max;
            preprocess_options_.blue_h_min = preprocess_cfg_default.blue_h_min;
            preprocess_options_.blue_h_max = preprocess_cfg_default.blue_h_max;
            preprocess_options_.blue_s_min = preprocess_cfg_default.blue_s_min;
            preprocess_options_.blue_s_max = preprocess_cfg_default.blue_s_max;
            preprocess_options_.blue_v_min = preprocess_cfg_default.blue_v_min;
            preprocess_options_.blue_v_max = preprocess_cfg_default.blue_v_max;
            preprocess_options_.binary_thresh = preprocess_cfg_default.binary_thresh;
            preprocess_options_.morph_type = preprocess_cfg_default.morph_type;
            preprocess_options_.binary_element = preprocess_cfg_default.binary_element;
            preprocess_options_.kernel_shape = preprocess_cfg_default.kernel_shape;
            preprocess_options_.kernel_w = preprocess_cfg_default.kernel_w;
            preprocess_options_.kernel_h = preprocess_cfg_default.kernel_h;
            preprocess_options_.kernel_angle_deg = preprocess_cfg_default.kernel_angle_deg;
            preprocess_options_.morph_iterations = preprocess_cfg_default.morph_iterations;

            const auto& draw_cfg_default = rm_radar_img_proc::DrawConfig::__getDefault__();
            draw_type_ = static_cast<DrawImage>(draw_cfg_default.draw_type);
            draw_options_.line_width = draw_cfg_default.line_width;
            show_fps_ = draw_cfg_default.show_fps;
            draw_options_.show_centroid_only = draw_cfg_default.show_centroid_only;
            show_all_armors_ = draw_cfg_default.show_all_armors;
        }

        auto armor_params_init = [this, &armor_nh]() {
            ROS_INFO("reading armor param");
            armor_nh.getParam("process_debug", process_debug_);
            detect_options_.process_debug = process_debug_;
            armor_nh.getParam("is_bar_debug", object_options_.bar_enabled);
            armor_nh.getParam("is_armor_debug", object_options_.armor_enabled);
            armor_nh.getParam("select_bar", select_bar_);
            armor_nh.getParam("enable_target_filter", detect_options_.enable_target_filter);
            armor_nh.getParam("save_yolo_samples", detect_options_.save_yolo_samples);
            armor_nh.getParam("yolo_refine_enabled", detect_options_.yolo_refine_enabled);
            armor_nh.getParam("max_angle_diff", object_options_.max_angle_diff);
            armor_nh.getParam("min_lw_ratio", object_options_.min_lw_ratio);
            armor_nh.getParam("max_lw_ratio", object_options_.max_lw_ratio);
            armor_nh.getParam("min_pixel_contained_ratio", object_options_.min_pixel_contained_ratio);
            armor_nh.getParam("max_bars_ratio", object_options_.max_bars_ratio);
            armor_nh.getParam("max_bars_distance", object_options_.max_bars_distance);
            armor_nh.getParam("min_bars_distance", object_options_.min_bars_distance);
            armor_nh.getParam("max_bars_angle", object_options_.max_bars_angle);
            armor_nh.getParam("max_bars_x_dis", object_options_.max_bars_x_dis);
            armor_nh.getParam("refine_max_brightness", detect_options_.refine_max_brightness);
            armor_nh.getParam("refine_roi_scale", detect_options_.refine_roi_scale);
            armor_nh.getParam("refine_search_start", detect_options_.refine_search_start);
            armor_nh.getParam("refine_search_end", detect_options_.refine_search_end);

            ROS_INFO("Armor params reading done");
        };
        auto pre_process_params_init = [this, &preprocess_nh]() {
            ROS_INFO("reading pre-process param");
            preprocess_nh.getParam("target_color", preprocess_options_.target_is_red);
            detect_options_.target_is_red = preprocess_options_.target_is_red;
            preprocess_nh.getParam("preprocess_method", preprocess_options_.preprocess_method);

            preprocess_nh.getParam("red_h_min_low", preprocess_options_.red_h_min_low);
            preprocess_nh.getParam("red_h_max_low", preprocess_options_.red_h_max_low);
            preprocess_nh.getParam("red_h_min_high", preprocess_options_.red_h_min_high);
            preprocess_nh.getParam("red_h_max_high", preprocess_options_.red_h_max_high);
            preprocess_nh.getParam("red_s_min", preprocess_options_.red_s_min);
            preprocess_nh.getParam("red_s_max", preprocess_options_.red_s_max);
            preprocess_nh.getParam("red_v_min", preprocess_options_.red_v_min);
            preprocess_nh.getParam("red_v_max", preprocess_options_.red_v_max);

            preprocess_nh.getParam("blue_h_min", preprocess_options_.blue_h_min);
            preprocess_nh.getParam("blue_h_max", preprocess_options_.blue_h_max);
            preprocess_nh.getParam("blue_s_min", preprocess_options_.blue_s_min);
            preprocess_nh.getParam("blue_s_max", preprocess_options_.blue_s_max);
            preprocess_nh.getParam("blue_v_min", preprocess_options_.blue_v_min);
            preprocess_nh.getParam("blue_v_max", preprocess_options_.blue_v_max);

            preprocess_nh.getParam("binary_thresh", preprocess_options_.binary_thresh);
            preprocess_nh.getParam("morph_type", preprocess_options_.morph_type);
            preprocess_nh.getParam("binary_element", preprocess_options_.binary_element);
            preprocess_nh.getParam("kernel_shape", preprocess_options_.kernel_shape);
            preprocess_nh.getParam("kernel_w", preprocess_options_.kernel_w);
            preprocess_nh.getParam("kernel_h", preprocess_options_.kernel_h);
            preprocess_nh.getParam("kernel_angle_deg", preprocess_options_.kernel_angle_deg);
            preprocess_nh.getParam("morph_iterations", preprocess_options_.morph_iterations);

            ROS_INFO("pre-processing param reading done");
        };
        auto draw_params_init = [this, &draw_nh]() {
            int draw_type_param = static_cast<int>(draw_type_);
            draw_nh.getParam("draw_type", draw_type_param);
            draw_type_ = static_cast<DrawImage>(draw_type_param);
            draw_nh.getParam("line_width", draw_options_.line_width);
            draw_nh.getParam("show_fps", show_fps_);
            draw_nh.getParam("show_centroid_only", draw_options_.show_centroid_only);
            draw_nh.getParam("show_all_armors", show_all_armors_);
        };

        armor_params_init();
        pre_process_params_init();
        draw_params_init();

        armor_cfg_srv_ = std::make_unique<dynamic_reconfigure::Server<rm_radar_img_proc::ArmorConfig>>(
            armor_cfg_mutex_, armor_nh);
        armor_cfg_cb_ = boost::bind(&Processor::armorconfigCB, this, _1, _2);
        armor_cfg_srv_->setCallback(armor_cfg_cb_);
        rm_radar_img_proc::ArmorConfig armor_cfg_init;
        armor_cfg_init.process_debug = process_debug_;
        armor_cfg_init.is_bar_debug = object_options_.bar_enabled;
        armor_cfg_init.is_armor_debug = object_options_.armor_enabled;
        armor_cfg_init.select_bar = select_bar_;
        armor_cfg_init.enable_target_filter = detect_options_.enable_target_filter;
        armor_cfg_init.save_yolo_samples = detect_options_.save_yolo_samples;
        armor_cfg_init.max_angle_diff = object_options_.max_angle_diff;
        armor_cfg_init.min_lw_ratio = object_options_.min_lw_ratio;
        armor_cfg_init.max_lw_ratio = object_options_.max_lw_ratio;
        armor_cfg_init.min_pixel_contained_ratio = object_options_.min_pixel_contained_ratio;
        armor_cfg_init.max_bars_ratio = object_options_.max_bars_ratio;
        armor_cfg_init.min_bars_distance = object_options_.min_bars_distance;
        armor_cfg_init.max_bars_distance = object_options_.max_bars_distance;
        armor_cfg_init.max_bars_angle = object_options_.max_bars_angle;
        armor_cfg_init.max_bars_x_dis = object_options_.max_bars_x_dis;
        armor_cfg_init.refine_max_brightness = detect_options_.refine_max_brightness;
        armor_cfg_init.refine_roi_scale = detect_options_.refine_roi_scale;
        armor_cfg_init.refine_search_start = detect_options_.refine_search_start;
        armor_cfg_init.refine_search_end = detect_options_.refine_search_end;
        armor_cfg_srv_->updateConfig(armor_cfg_init);

        preprocess_cfg_srv_ = std::make_unique<dynamic_reconfigure::Server<rm_radar_img_proc::PreprocessConfig>>(
            preprocess_cfg_mutex_, preprocess_nh);
        preprocess_cfg_cb_ = boost::bind(&Processor::preProcessconfigCB, this, _1, _2);
        preprocess_cfg_srv_->setCallback(preprocess_cfg_cb_);
        rm_radar_img_proc::PreprocessConfig preprocess_cfg_init;
        preprocess_cfg_init.target_color = preprocess_options_.target_is_red;
        preprocess_cfg_init.preprocess_method = preprocess_options_.preprocess_method;
        preprocess_cfg_init.red_h_min_low = preprocess_options_.red_h_min_low;
        preprocess_cfg_init.red_h_max_low = preprocess_options_.red_h_max_low;
        preprocess_cfg_init.red_h_min_high = preprocess_options_.red_h_min_high;
        preprocess_cfg_init.red_h_max_high = preprocess_options_.red_h_max_high;
        preprocess_cfg_init.red_s_min = preprocess_options_.red_s_min;
        preprocess_cfg_init.red_s_max = preprocess_options_.red_s_max;
        preprocess_cfg_init.red_v_min = preprocess_options_.red_v_min;
        preprocess_cfg_init.red_v_max = preprocess_options_.red_v_max;
        preprocess_cfg_init.blue_h_min = preprocess_options_.blue_h_min;
        preprocess_cfg_init.blue_h_max = preprocess_options_.blue_h_max;
        preprocess_cfg_init.blue_s_min = preprocess_options_.blue_s_min;
        preprocess_cfg_init.blue_s_max = preprocess_options_.blue_s_max;
        preprocess_cfg_init.blue_v_min = preprocess_options_.blue_v_min;
        preprocess_cfg_init.blue_v_max = preprocess_options_.blue_v_max;
        preprocess_cfg_init.binary_thresh = preprocess_options_.binary_thresh;
        preprocess_cfg_init.morph_type = preprocess_options_.morph_type;
        preprocess_cfg_init.binary_element = preprocess_options_.binary_element;
        preprocess_cfg_init.kernel_shape = preprocess_options_.kernel_shape;
        preprocess_cfg_init.kernel_w = preprocess_options_.kernel_w;
        preprocess_cfg_init.kernel_h = preprocess_options_.kernel_h;
        preprocess_cfg_init.kernel_angle_deg = preprocess_options_.kernel_angle_deg;
        preprocess_cfg_init.morph_iterations = preprocess_options_.morph_iterations;
        preprocess_cfg_srv_->updateConfig(preprocess_cfg_init);

        draw_cfg_srv_ = std::make_unique<dynamic_reconfigure::Server<rm_radar_img_proc::DrawConfig>>(
            draw_cfg_mutex_, draw_nh);
        draw_cfg_cb_ = boost::bind(&Processor::drawconfigCB, this, _1, _2);
        draw_cfg_srv_->setCallback(draw_cfg_cb_);
        rm_radar_img_proc::DrawConfig draw_cfg_init;
        draw_cfg_init.draw_type = static_cast<int>(draw_type_);
        draw_cfg_init.line_width = draw_options_.line_width;
        draw_cfg_init.show_fps = show_fps_;
        draw_cfg_init.show_centroid_only = draw_options_.show_centroid_only;
        draw_cfg_init.show_all_armors = show_all_armors_;
        draw_cfg_srv_->updateConfig(draw_cfg_init);
    }

    //dynamic reconfigure callback
    void Processor::armorconfigCB(rm_radar_img_proc::ArmorConfig& config, uint32_t level)
    {
        std::lock_guard<std::mutex> lock(process_mutex_);
        ROS_INFO("reading Armor dynamic reconfigure callback");
        process_debug_ = config.process_debug;
        detect_options_.process_debug = process_debug_;
        object_options_.bar_enabled = config.is_bar_debug;
        object_options_.armor_enabled = config.is_armor_debug;
        select_bar_ = config.select_bar;
        detect_options_.enable_target_filter = config.enable_target_filter;
        detect_options_.save_yolo_samples = config.save_yolo_samples;
        object_options_.max_angle_diff = config.max_angle_diff;
        object_options_.min_lw_ratio = config.min_lw_ratio;
        object_options_.max_lw_ratio = config.max_lw_ratio;
        object_options_.min_pixel_contained_ratio = config.min_pixel_contained_ratio;
        object_options_.max_bars_ratio = config.max_bars_ratio;
        object_options_.max_bars_distance = config.max_bars_distance;
        object_options_.min_bars_distance = config.min_bars_distance;
        object_options_.max_bars_angle = config.max_bars_angle;
        object_options_.max_bars_x_dis = config.max_bars_x_dis;
        detect_options_.refine_max_brightness = config.refine_max_brightness;
        detect_options_.refine_roi_scale = config.refine_roi_scale;
        detect_options_.refine_search_start = config.refine_search_start;
        detect_options_.refine_search_end = config.refine_search_end;

        detect_options_.select_bar = select_bar_;
        detect_options_.object_options = object_options_;
        if (detect_core_)
        {
            detect_core_->setOptions(detect_options_);
        }

        if (!armor_dynamic_reconfig_initialized_)
        {
            armor_dynamic_reconfig_initialized_ = true;
        }

        ROS_INFO("Armor dynamic reconfigure callback done");
    }

    void Processor::preProcessconfigCB(rm_radar_img_proc::PreprocessConfig& config, uint32_t level)
    {
        std::lock_guard<std::mutex> lock(process_mutex_);
        preprocess_options_.target_is_red = config.target_color;
        detect_options_.target_is_red = preprocess_options_.target_is_red;
        preprocess_options_.preprocess_method = config.preprocess_method;

        preprocess_options_.red_h_min_low = config.red_h_min_low;
        preprocess_options_.red_h_max_low = config.red_h_max_low;
        preprocess_options_.red_h_min_high = config.red_h_min_high;
        preprocess_options_.red_h_max_high = config.red_h_max_high;
        preprocess_options_.red_s_min = config.red_s_min;
        preprocess_options_.red_s_max = config.red_s_max;
        preprocess_options_.red_v_min = config.red_v_min;
        preprocess_options_.red_v_max = config.red_v_max;

        preprocess_options_.blue_h_min = config.blue_h_min;
        preprocess_options_.blue_h_max = config.blue_h_max;
        preprocess_options_.blue_s_min = config.blue_s_min;
        preprocess_options_.blue_s_max = config.blue_s_max;
        preprocess_options_.blue_v_min = config.blue_v_min;
        preprocess_options_.blue_v_max = config.blue_v_max;

        preprocess_options_.binary_thresh = config.binary_thresh;

        preprocess_options_.morph_type = config.morph_type;
        preprocess_options_.binary_element = config.binary_element;
        preprocess_options_.kernel_shape = config.kernel_shape;
        preprocess_options_.kernel_w = config.kernel_w;
        preprocess_options_.kernel_h = config.kernel_h;
        preprocess_options_.kernel_angle_deg = config.kernel_angle_deg;
        preprocess_options_.morph_iterations = config.morph_iterations;

        if (!pre_process_dynamic_reconfig_initialized_)
        {
            pre_process_dynamic_reconfig_initialized_ = true;
        }

        if (preprocess_core_)
        {
            preprocess_core_->setOptions(preprocess_options_);
            preprocess_options_ = preprocess_core_->options();
        }

        detect_options_.target_is_red = preprocess_options_.target_is_red;

        if (detect_core_)
        {
            detect_core_->setOptions(detect_options_);
        }
    }


    void Processor::drawconfigCB(rm_radar_img_proc::DrawConfig& config, uint32_t level)
    {
        std::lock_guard<std::mutex> lock(process_mutex_);
        const int draw_type_clamped = std::max(static_cast<int>(DrawImage::DISABLE),
                                               std::min(config.draw_type, static_cast<int>(DrawImage::PROJECT)));
        draw_type_ = static_cast<DrawImage>(draw_type_clamped);
        draw_options_.line_width = config.line_width;
        draw_options_.show_centroid_only = config.show_centroid_only;
        show_all_armors_ = config.show_all_armors;

        if (show_fps_ != config.show_fps)
        {
            show_fps_ = config.show_fps;
            if (visualizer_)
            {
                visualizer_->resetFps();
            }
        }
        else
        {
            show_fps_ = config.show_fps;
        }

        if (visualizer_)
        {
            visualizer_->setOptions(draw_options_);
        }
    }

    void Processor::processingLoop()
    {
        while (true)
        {
            cv_bridge::CvImageConstPtr next_frame_cv_ptr;
            std_msgs::Header frame_header;
            {
                std::unique_lock<std::mutex> lock(frame_mutex_);
                frame_cv_.wait(lock, [this]() { return stop_worker_ || has_pending_frame_; });
                if (stop_worker_)
                {
                    return;
                }

                next_frame_cv_ptr.swap(latest_frame_cv_ptr_);
                frame_header = latest_frame_header_;
                has_pending_frame_ = false;
            }

            if (!next_frame_cv_ptr || next_frame_cv_ptr->image.empty())
            {
                continue;
            }

            std::lock_guard<std::mutex> lock(process_mutex_);
            if (!camera_model_initialized_)
            {
                continue;
            }

            target_header_ = frame_header;
            binary_image_ = nullptr;
            morphology_image_ = nullptr;
            const cv::Mat& frame_bgr = next_frame_cv_ptr->image;
            if (detect_options_.detect_method == DetectMethod::YOLO)
            {
                detect(frame_bgr);
                publishDetection();
                draw(frame_bgr);
                continue;
            }

            cv::UMat frame = frame_bgr.getUMat(cv::ACCESS_READ);
            preprocess(frame);
            detect(frame);
            publishDetection();
            draw(frame);
        }
    }

    void Processor::preprocess(const cv::UMat& frame)
    {
        if (detect_options_.detect_method == DetectMethod::YOLO)
        {
            return;
        }

        preprocess_core_->process(frame);
        binary_image_ = &preprocess_core_->getBinaryImage();
        morphology_image_ = &preprocess_core_->getMorphologyImage();
    }

    void Processor::detect(const cv::Mat& frame)
    {
        detect_core_->detect(frame);

        bars_ = &detect_core_->getBars();
        debug_armors_ = &detect_core_->getDebugArmors();
        best_armor_ = detect_core_->getBestArmor();
    }

    void Processor::detect(const cv::UMat& frame)
    {
        if (detect_options_.detect_method == DetectMethod::YOLO)
        {
            detect_core_->detect(frame);
        }
        else
        {
            detect_core_->detect(frame, morphology_image_);
        }

        bars_ = &detect_core_->getBars();
        debug_armors_ = &detect_core_->getDebugArmors();
        best_armor_ = detect_core_->getBestArmor();
    }

    void Processor::draw(const cv::Mat& frame)
    {
        if (draw_type_ == DrawImage::DISABLE)
        {
            return;
        }

        cv::Mat draw_image;
        sensor_msgs::ImagePtr msg;
        const auto draw_selected_armors = [this](cv::Mat& image) {
            if (show_all_armors_)
            {
                visualizer_->draw(image, *debug_armors_);
            }
            else if (best_armor_)
            {
                visualizer_->draw(image, *best_armor_);
            }
        };
        const auto draw_selected_vertices = [this](cv::Mat& image) {
            if (show_all_armors_)
            {
                visualizer_->drawVertexes(image, *debug_armors_, image_center_);
            }
            else if (best_armor_)
            {
                visualizer_->drawVertexes(image, *best_armor_, image_center_);
            }
        };
        switch (draw_type_)
        {
            case DrawImage::RAW:
                frame.copyTo(draw_image);
                break;
            case DrawImage::BINARY:
                if (binary_image_)
                {
                    draw_image = binary_image_->getMat(cv::ACCESS_READ).clone();
                }
                break;
            case DrawImage::MORPHOLOGY:
                if (morphology_image_)
                {
                    cv::cvtColor(morphology_image_->getMat(cv::ACCESS_READ), draw_image, cv::COLOR_GRAY2BGR);
                }
                break;
            case DrawImage::BARS:
                frame.copyTo(draw_image);
                visualizer_->draw(draw_image, *bars_);
                break;
            case DrawImage::ARMORS:
                frame.copyTo(draw_image);
                draw_selected_armors(draw_image);
                break;
            case DrawImage::ARMORS_VERTEXS:
                frame.copyTo(draw_image);
                draw_selected_vertices(draw_image);
                break;
            case DrawImage::BARS_ARMORS:
                frame.copyTo(draw_image);
                draw_selected_armors(draw_image);
                visualizer_->draw(draw_image, *bars_);
                break;
            default:
                frame.copyTo(draw_image);
                draw_selected_armors(draw_image);
                visualizer_->draw(draw_image, *bars_);
                break;
        }
        if (show_fps_ && !draw_image.empty())
        {
            visualizer_->showFps(draw_image);
        }
        if (!draw_image.empty())
        {
            const bool is_mono = (draw_type_ == DrawImage::BINARY);
            msg = cv_bridge::CvImage(std_msgs::Header(), is_mono ? "mono8" : "bgr8", draw_image).toImageMsg();
        }
        if (msg)
        {
            image_pub_.publish(msg);
        }
    }

    void Processor::draw(const cv::UMat& frame)
    {
        if (draw_type_ == DrawImage::DISABLE)
        {
            return;
        }

        cv::UMat draw_image;
        sensor_msgs::ImagePtr msg;
        const auto draw_selected_armors = [this](cv::UMat& image) {
            if (show_all_armors_)
            {
                visualizer_->draw(image, *debug_armors_);
            }
            else if (best_armor_)
            {
                visualizer_->draw(image, *best_armor_);
            }
        };
        const auto draw_selected_vertices = [this](cv::UMat& image) {
            if (show_all_armors_)
            {
                visualizer_->drawVertexes(image, *debug_armors_, image_center_);
            }
            else if (best_armor_)
            {
                visualizer_->drawVertexes(image, *best_armor_, image_center_);
            }
        };
        switch (draw_type_)
            {
                case DrawImage::RAW:
                    frame.copyTo(draw_image);
                    break;
                case DrawImage::BINARY:
                    if (binary_image_) binary_image_->copyTo(draw_image);
                    break;
                case DrawImage::MORPHOLOGY:
                    if (morphology_image_) cv::cvtColor(*morphology_image_, draw_image, cv::COLOR_GRAY2BGR);
                    break;
                case DrawImage::BARS:
                    frame.copyTo(draw_image);
                    visualizer_->draw(draw_image, *bars_);
                    break;
                case DrawImage::ARMORS:
                    frame.copyTo(draw_image);
                    draw_selected_armors(draw_image);
                    break;
                case DrawImage::ARMORS_VERTEXS:
                    frame.copyTo(draw_image);
                    draw_selected_vertices(draw_image);
                    break;
                case DrawImage::BARS_ARMORS:
                    frame.copyTo(draw_image);
                    draw_selected_armors(draw_image);
                    visualizer_->draw(draw_image, *bars_);
                    break;
                default:
                    frame.copyTo(draw_image);
                    draw_selected_armors(draw_image);
                    visualizer_->draw(draw_image, *bars_);
                    break;
            }
        if (show_fps_ && !draw_image.empty())
        {
            visualizer_->showFps(draw_image);
        }
        if (!draw_image.empty())
        {
            const bool is_mono = (draw_type_ == DrawImage::BINARY);
            msg = cv_bridge::CvImage(
                std_msgs::Header(),
                is_mono ? "mono8" : "bgr8",
                draw_image.getMat(cv::ACCESS_READ)).toImageMsg();
        }
        if(msg)
        {
            image_pub_.publish(msg);
        }
    }
    void Processor::publishDetection()
    {
        if (!best_armor_)
        {
            consecutive_detection_count_ = 0;
            if (process_debug_)
            {
                ROS_INFO_THROTTLE(1.0, "[t=%.3f][Processor] No target selected in current frames, skip publish.",
                                  target_header_.stamp.toSec());
            }
            return;
        }

        const auto& armor = *best_armor_;
        const auto& armor_points = armor.bars_4points_;
        if (armor_points.size() != 4)
        {
            consecutive_detection_count_ = 0;
            ROS_WARN_THROTTLE(2, "[publishDetection] Invalid armor point count: %lu", armor_points.size());
            return;
        }

        if (consecutive_detection_count_ < kDetectionConfirmFrames)
        {
            ++consecutive_detection_count_;
        }
        if (consecutive_detection_count_ < kDetectionConfirmFrames)
        {
            if (process_debug_)
            {
                ROS_INFO_THROTTLE(1.0,
                                  "[t=%.3f][Processor] Detection warmup %d/%d, skip publish.",
                                  target_header_.stamp.toSec(),
                                  consecutive_detection_count_,
                                  kDetectionConfirmFrames);
            }
            return;
        }

        const cv::Point2d& center = armor.center_;

        const double distance2ImgCenter = cv::norm(center - image_center_);
        const double raw_error_x = image_center_.x - center.x;
        const double raw_error_y = image_center_.y - center.y;

        rm_radar_msgs::DroneDetection target;
        target.header = target_header_;
        target.is_cam_msg = true;
        target.is_lidar_msg = false;
        target.confidence = armor.confidence_;
        target.distance_to_image_center = distance2ImgCenter;

        for (size_t i = 0; i < 4; i++)
        {
            target.armor_points[i].x = armor_points[i].x;
            target.armor_points[i].y = armor_points[i].y;
        }

        target.target_centroid_x = static_cast<uint32_t>(center.x);
        target.target_centroid_y = static_cast<uint32_t>(center.y);
        target.error_x = raw_error_x;
        target.error_y = raw_error_y;
        target.error_angle_yaw = std::atan(target.error_x / fx_);
        target.error_angle_pitch = std::atan(target.error_y / fy_);

        target_pub_single_.publish(target);
        if (process_debug_)
        {
            ROS_INFO_THROTTLE(1.0,
                              "[t=%.3f][Processor] Published target center=(%.1f, %.1f) conf=%.3f err=(%.1f, %.1f)",
                              target_header_.stamp.toSec(), center.x, center.y, armor.confidence_, raw_error_x, raw_error_y);
        }
    }



}  // namespace rm_radar_imgProc
