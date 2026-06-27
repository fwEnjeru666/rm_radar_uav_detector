#pragma once

#include <algorithm>
#include <vector>
#include <array>
#include <cmath>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <numeric>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <nodelet/nodelet.h>
#include <boost/thread/recursive_mutex.hpp>
#include <rm_radar_msgs/DroneDetection.h>
#include <std_msgs/Header.h>
#include <dynamic_reconfigure/server.h>
#include <rm_radar_img_proc/PreprocessConfig.h>
#include <rm_radar_img_proc/ArmorConfig.h>
#include <rm_radar_img_proc/DrawConfig.h>
#include <armor.h>
#include <object_options.h>
#include <tools.h>
#include <preprocess.h>
#include <detect.h>
#include <common.h>
#include <cstdio>


namespace rm_radarplugin
{
    class Processor : public nodelet::Nodelet
    {

    public:
        Processor(){};
        ~Processor(){
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                stop_worker_ = true;
            }
            frame_cv_.notify_all();
            cam_sub_.shutdown();
            callback_queue_.disable();
            callback_queue_.clear();
            if (worker_thread_.joinable())
                worker_thread_.join();
            if (my_thread_.joinable())
                my_thread_.join();
        };
        void onInit();
        void armorconfigCB(rm_radar_img_proc::ArmorConfig &config, uint32_t level);
        void drawconfigCB(rm_radar_img_proc::DrawConfig &config, uint32_t level);
        void preProcessconfigCB(rm_radar_img_proc::PreprocessConfig &config, uint32_t level);
        void initialize(ros::NodeHandle &nh);


        void preprocess(const cv::UMat& frame);
        void detect(const cv::Mat& frame);
        void detect(const cv::UMat& frame);
        void publishDetection();

        ///draw
        void draw(const cv::Mat& frame);
        void draw(const cv::UMat& frame);

    private:
        void setDynamicReconfig();
        void processingLoop();

        std::thread my_thread_;
        std::thread worker_thread_;
        ros::NodeHandle nh_;
        ros::CallbackQueue callback_queue_;

        //
        std::shared_ptr<image_transport::ImageTransport> it_;
        image_transport::CameraSubscriber cam_sub_;
        image_transport::Publisher image_pub_;

        cv::Mat intrinsics_;
        cv::Mat dist_coeffs_;
        double fx_ ;
        double fy_ ;
        double cx_ ; // 真实的图像中心 x
        double cy_ ; // 真实的图像中心 
        cv::Point2d image_center_; // 计算得到的图像中心
        std_msgs::Header target_header_{};
        std::mutex frame_mutex_;
        std::condition_variable frame_cv_;
        cv_bridge::CvImageConstPtr latest_frame_cv_ptr_{};
        std_msgs::Header latest_frame_header_{};
        bool has_pending_frame_{false};
        bool stop_worker_{false};
        std::mutex process_mutex_;
        
        void cam_callback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info)
        {
            const auto temp = cv_bridge::toCvShare(img, "bgr8");
            cv_bridge::CvImageConstPtr stale_frame_cv_ptr;

            if (!camera_model_initialized_)
            {
                std::lock_guard<std::mutex> process_lock(process_mutex_);
                if (!camera_model_initialized_)
                {
                    intrinsics_ = cv::Mat(3, 3, CV_64F, (void*)info->K.data()).clone();
                    dist_coeffs_ = cv::Mat(info->D).clone();
                    camera_model_initialized_ = true;
                    fx_ = intrinsics_.at<double>(0, 0);
                    fy_ = intrinsics_.at<double>(1, 1);
                    cx_ = intrinsics_.at<double>(0, 2);
                    cy_ = intrinsics_.at<double>(1, 2);
                    image_center_ = cv::Point2d(cx_, cy_);
                }
            }

            {
                std::lock_guard<std::mutex> frame_lock(frame_mutex_);
                stale_frame_cv_ptr.swap(latest_frame_cv_ptr_);
                latest_frame_cv_ptr_ = temp;
                latest_frame_header_ = info->header;
                has_pending_frame_ = true;
            }
            frame_cv_.notify_one();
        }

        /// bar compute
        // int bar_br_thresh_{};
        int select_bar_{};
        /// id classification
        std::string xml_path_{};
        std::string bin_path_{};
        std::pair<int, float> result_{};
        double negative_confidence_{};
        double firstnet_score_;

        /// draw
        DrawImage draw_type_{};
        bool show_all_armors_{false};
        bool process_debug_{false};

        // Show FPS 
        bool show_fps_{false};

        ObjectOptions object_options_{};
        tools::VisualizerOptions draw_options_{};
        std::unique_ptr<DetectCore> detect_core_{};
        const std::vector<Bar>* bars_{nullptr};
        const std::vector<Armor>* debug_armors_{nullptr};
        Armor* best_armor_{nullptr};
        std::unique_ptr<tools::Visualizer> visualizer_{};

        PreprocessOptions preprocess_options_{};
        std::unique_ptr<PreprocessCore> preprocess_core_{};
        const cv::UMat* binary_image_{nullptr};
        const cv::UMat* morphology_image_{nullptr};


        ///debuger
        DetectOptions detect_options_{};
        //dynamic reconfig
        std::unique_ptr<dynamic_reconfigure::Server<rm_radar_img_proc::PreprocessConfig>> preprocess_cfg_srv_;
        dynamic_reconfigure::Server<rm_radar_img_proc::PreprocessConfig>::CallbackType preprocess_cfg_cb_;
        boost::recursive_mutex preprocess_cfg_mutex_;
        bool pre_process_dynamic_reconfig_initialized_ = false;

        std::unique_ptr<dynamic_reconfigure::Server<rm_radar_img_proc::ArmorConfig>> armor_cfg_srv_;
        dynamic_reconfigure::Server<rm_radar_img_proc::ArmorConfig>::CallbackType armor_cfg_cb_;
        boost::recursive_mutex armor_cfg_mutex_;
        bool armor_dynamic_reconfig_initialized_ = false;

        std::unique_ptr<dynamic_reconfigure::Server<rm_radar_img_proc::DrawConfig>> draw_cfg_srv_;
        dynamic_reconfigure::Server<rm_radar_img_proc::DrawConfig>::CallbackType draw_cfg_cb_;
        boost::recursive_mutex draw_cfg_mutex_;
        ///

        ///
        ros::Publisher target_pub_single_{};
        bool camera_model_initialized_{false};
        int consecutive_detection_count_{0};
        static constexpr int kDetectionConfirmFrames = 5;

    };
}
