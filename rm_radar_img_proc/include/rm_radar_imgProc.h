#pragma once

#include <iostream>
#include <vector>
#include <dirent.h>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <cmath>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <mutex>
#include <numeric>
#include <thread>
#include <nodelet/nodelet.h>
#include <pluginlib/class_loader.h>
#include <pluginlib/class_list_macros.h>
#include <rm_radar_msgs/DroneDetectionArray.h>
#include <rm_radar_msgs/DroneDetection.h>
#include <rm_radar_msgs/DroneTrackData.h>
#include <rm_msgs/TrackData.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <dynamic_reconfigure/server.h>
#include <rm_radar_img_proc/PreprocessConfig.h>
#include <rm_radar_img_proc/ArmorConfig.h>
#include <rm_radar_img_proc/DrawConfig.h>
#include <rm_vision/vision_base/processor_interface.h>
#include <common.h>



namespace rm_radarplugin
{
    class Bar
    {
        public:
        cv::RotatedRect bar_rect_;
        double length_len_;  /// long side
        double width_len_;   /// short side
        double lw_ratio_;
        cv::Point2f points_[4];
        cv::Point2f center_point_;
        double angle_;  /// the angle between short side of bar and x axis
        double pixel_contained_ratio_;
        ArmorColor color_;
        bool is_clockwise_;
        int used_num_;
        std::vector<cv::Point> contour_;

        Bar(cv::RotatedRect bar_rect, const std::vector<cv::Point>& contour, ArmorColor color) : bar_rect_(std::move(bar_rect))
        {
            length_len_ = std::max(bar_rect_.size.width, bar_rect_.size.height);
            width_len_ = std::min(bar_rect_.size.width, bar_rect_.size.height);
            pixel_contained_ratio_ = cv::contourArea(contour) / bar_rect_.size.area();
            lw_ratio_ = length_len_ / width_len_;
            bar_rect_.points(points_);
            center_point_ = bar_rect_.center;
            contour_ = contour;

            if (length_len_ == bar_rect_.size.width)
            {
            angle_ = bar_rect_.angle + 90.0;
            is_clockwise_ = true;
            }
            else
            {
            angle_ = bar_rect_.angle;  
            is_clockwise_ = false;
            }
            color_ = color;
        };
        };

    class Armor
        {
        public:
        Bar* bar_up_;
        Bar* bar_bottom_;
        std::vector<cv::Point2d> bars_4points_;        /// bl, tl, tr, br ; center point
        std::vector<cv::Point2d> bars_inter_4points_;  /// bl, tl, tr, br ; inter point
        std::vector<cv::Point2d> bars_PCA_4points_;
        float warp_white_ratio_;
        double length;
        double width;
        cv::Point2d center_;
        double lw_rate_;
        int id_;
        double confidence_;
        double negative_confidence_;
        double area_;
        double parallel_dist_;
        double vertical_dist_;

        Armor(Bar& bar_up, Bar& bar_bottom)
        {
            bar_up_ = &bar_up;
            bar_bottom_ = &bar_bottom;
            getBarsPoints();
            getArmorArea();
            getCenterPoint();
            getParallelDist();
            getVerticalDist();
            id_ = 0;
            confidence_ = 0;
        };

        std::vector<cv::Point2f> getHorizontalBarEndpoints(const Bar* bar)
        {
            cv::Point2f pts[4];
            bar->bar_rect_.points(pts);

            std::sort(pts, pts + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.x < b.x;
            });

            cv::Point2f l_c, r_c; //left center, right center

            l_c = (pts[0] + pts[1]) * 0.5;
            r_c = (pts[2] + pts[3]) * 0.5;
            return {l_c, r_c};

        }

        void getBarsPoints()
        {
            bars_4points_.clear();

            //(TL -> TR -> BR -> BL)
            std::vector<cv::Point2f> up_bar_pts = getHorizontalBarEndpoints(bar_up_);
            cv::Point2f top_left = up_bar_pts[0];
            cv::Point2f top_right = up_bar_pts[1];


            std::vector<cv::Point2f> bottom_bar_pts = getHorizontalBarEndpoints(bar_bottom_);
            cv::Point2f bottom_left = bottom_bar_pts[0];
            cv::Point2f bottom_right = bottom_bar_pts[1];

            //p0: TL
            bars_4points_.emplace_back(top_left);
            //p1: TR
            bars_4points_.emplace_back(top_right);
            //p2: BR
            bars_4points_.emplace_back(bottom_right);
            //p3: BL
            bars_4points_.emplace_back(bottom_left);
        };

        void getArmorArea()
        {
            double len1 =
                pow(pow(bars_4points_[0].x - bars_4points_[1].x, 2) + pow(bars_4points_[0].y - bars_4points_[1].y, 2), 0.5);
            double len2 =
                pow(pow(bars_4points_[1].x - bars_4points_[2].x, 2) + pow(bars_4points_[1].y - bars_4points_[2].y, 2), 0.5);
            double len3 =
                pow(pow(bars_4points_[0].x - bars_4points_[2].x, 2) + pow(bars_4points_[0].y - bars_4points_[2].y, 2), 0.5);
            double len4 =
                pow(pow(bars_4points_[2].x - bars_4points_[3].x, 2) + pow(bars_4points_[2].y - bars_4points_[3].y, 2), 0.5);
            double len5 =
                pow(pow(bars_4points_[0].x - bars_4points_[3].x, 2) + pow(bars_4points_[0].y - bars_4points_[3].y, 2), 0.5);

            double half_cir1 = (len1 + len2 + len3) / 2.0;
            double half_cir2 = (len3 + len4 + len5) / 2.0;

            double area1 = pow((half_cir1 * (half_cir1 - len1) * (half_cir1 - len2) * (half_cir1 - len3)), 0.5);
            double area2 = pow((half_cir2 * (half_cir2 - len3) * (half_cir2 - len4) * (half_cir2 - len5)), 0.5);

            area_ = area1 + area2;
        };

        void getCenterPoint()
        {
            double p1_x = bars_4points_[0].x;
            double p1_y = bars_4points_[0].y;
            double p2_x = bars_4points_[1].x;
            double p2_y = bars_4points_[1].y;
            double p3_x = bars_4points_[2].x;
            double p3_y = bars_4points_[2].y;
            double p4_x = bars_4points_[3].x;
            double p4_y = bars_4points_[3].y;

            double line1_k = (p1_y - p3_y) / (p1_x - p3_x + 0.000000001);
            double line2_k = (p2_y - p4_y) / (p2_x - p4_x + 0.000000001);
            double line1_b = ((p1_y + p3_y) - line1_k * (p1_x + p3_x)) / 2;
            double line2_b = ((p2_y + p4_y) - line2_k * (p2_x + p4_x)) / 2;

            double cross_point_x = (line2_b - line1_b) / (line1_k - line2_k + 0.000000001);
            double cross_point_y = ((line1_k + line2_k) * cross_point_x + line1_b + line2_b) * 0.5;

            center_.x = int(cross_point_x);
            center_.y = int(cross_point_y);
        }

        void getParallelDist()
        {
            double p1_x = bars_4points_[0].x;
            double p1_y = bars_4points_[0].y;
            double p2_x = bars_4points_[1].x;
            double p2_y = bars_4points_[1].y;
            double p3_x = bars_4points_[2].x;
            double p3_y = bars_4points_[2].y;
            double p4_x = bars_4points_[3].x;
            double p4_y = bars_4points_[3].y;

            double line1_k = (p1_y - p2_y) / (p1_x - p2_x + 0.000000001);
            double line2_k = (p3_y - p4_y) / (p3_x - p4_x + 0.000000001);
            double line1_b = ((p1_y + p2_y) - line1_k * (p1_x + p2_x)) / 2;
            double line2_b = ((p3_y + p4_y) - line2_k * (p3_x + p4_x)) / 2;

            double dist = pow(pow(bar_up_->center_point_.x - bar_bottom_->center_point_.x, 2) +
                                pow(bar_up_->center_point_.y - bar_bottom_->center_point_.y, 2),
                            0.5);
            double vertical_dist1 = fabs(line1_k * bar_bottom_->center_point_.x - bar_bottom_->center_point_.y + line1_b) /
                                    pow(pow(line1_k, 2) + 1, 0.5);
            double parallel_dist1 = pow(pow(dist, 2) - pow(vertical_dist1, 2), 0.5);

            double vertical_dist2 = fabs(line2_k * bar_up_->center_point_.x - bar_up_->center_point_.y + line2_b) /
                                    pow(pow(line2_k, 2) + 1, 0.5);
            double parallel_dist2 = pow(pow(dist, 2) - pow(vertical_dist2, 2), 0.5);

            parallel_dist_ = (parallel_dist1 + parallel_dist2) * 0.5;
        }

        void getVerticalDist()
        {

            double p1_x = bars_4points_[0].x;
            double p1_y = bars_4points_[0].y;
            double p2_x = bars_4points_[1].x;
            double p2_y = bars_4points_[1].y;
            
            double p3_x = bars_4points_[2].x;
            double p3_y = bars_4points_[2].y;
            double p4_x = bars_4points_[3].x;
            double p4_y = bars_4points_[3].y;

            double line_top_k = (p1_y - p2_y) / (p1_x - p2_x + 1e-9);
            double line_bot_k = (p3_y - p4_y) / (p3_x - p4_x + 1e-9);

            double line_top_b = ((p1_y + p2_y) - line_top_k * (p1_x + p2_x)) / 2.0;
            double line_bot_b = ((p3_y + p4_y) - line_bot_k * (p3_x + p4_x)) / 2.0;

            double center_top_x = (p1_x + p2_x) / 2.0;
            double center_top_y = (p1_y + p2_y) / 2.0;
            double center_bot_x = (p3_x + p4_x) / 2.0;
            double center_bot_y = (p3_y + p4_y) / 2.0;
            double dist_bot_to_top_line = fabs(line_top_k * center_bot_x - center_bot_y + line_top_b) /
                                        sqrt(pow(line_top_k, 2) + 1);


            double dist_top_to_bot_line = fabs(line_bot_k * center_top_x - center_top_y + line_bot_b) /
                                        sqrt(pow(line_bot_k, 2) + 1);


            double vertical_dist = (dist_bot_to_top_line + dist_top_to_bot_line) / 2.0;
            
            vertical_dist_ = vertical_dist;
        }

    };
    class Processor : public rm_vision::ProcessorInterface , public nodelet::Nodelet
    {
    struct MatchPairs
    {
        size_t index_top;
        size_t index_bottom;
        double score;
        
        MatchPairs(size_t top, size_t bottom, double s)
            : index_top(top), index_bottom(bottom), score(s) {}
    };

    public:
        Processor(){};
        ~Processor(){
            if (my_thread_.joinable())
                my_thread_.join();
        };
        void onInit();
        void armorconfigCB(rm_radar_img_proc::ArmorConfig &config, uint32_t level);
        void drawconfigCB(rm_radar_img_proc::DrawConfig &config, uint32_t level);
        void preProcessconfigCB(rm_radar_img_proc::PreprocessConfig &config, uint32_t level);
        void initialize(ros::NodeHandle &nh) override;


        cv::Mat setElement();
        void hsv2Binary();
        void bgr2Binary();
        void imageProcess(cv_bridge::CvImagePtr &cv_image) override;

        void findbars();
        bool isValidBar(const Bar& bar);
        double getArmorScore(const Bar& bar_top, const Bar& bar_bottom);
        void findArmor() override;
        bool isValidArmor(Bar& top_bar, Bar& bottom_bar);
        void paramReconfig() override;

        ///draw
        void drawBars(cv::Mat& image);
        void drawArmors(cv::Mat& image, std::vector<Armor>& armors);
        void drawArmors(cv::Mat& image) { drawArmors(image, armors_); }
        void drawArmorsVertexes(cv::Mat& image, std::vector<Armor>& armors);
        void drawArmorsVertexes(cv::Mat& image) { drawArmorsVertexes(image, armors_); }
        bool drawWarp();
        void drawTracker(cv::Mat& image);
        void draw() override;

        // tracker callback
        void trackerCB(const rm_msgs::TrackData::ConstPtr& msg);
        // project 3D point to 2D image
        bool projectPoint3Dto2D(const geometry_msgs::Point& point_3d, cv::Point2d& point_2d);

        
        rm_vision::ProcessorInterface::Object getObj() override;
        void putObj() override;

    private:
        std::thread my_thread_;
        ros::NodeHandle nh_;

        //
        std::shared_ptr<image_transport::ImageTransport> it_;
        image_transport::CameraSubscriber tele_cam_sub_;
        image_transport::CameraSubscriber wide_cam_sub_;
        image_transport::Publisher image_pub_;

        cv::Mat intrinsics_;
        cv::Mat dist_coeffs_;
        sensor_msgs::CameraInfoConstPtr camera_info_;

        //cam_mode
        bool is_tele_cam_{true}; // true: tele_cam, false: wide_cam
        void tele_cam_callback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info)
        {
            if(!is_tele_cam_) return;
            ROS_INFO_ONCE("[cam_mode] tele camera mode");
            if (!target_is_armor_)
            {
            //      ROS_INFO("not armor");
            return;
            }
            camera_info_ = info;
            target_array_.header = info->header;
            intrinsics_ = cv::Mat(3, 3, CV_64F, (void*)info->K.data()).clone();
            dist_coeffs_ = cv::Mat(info->D).clone();
            boost::shared_ptr<cv_bridge::CvImage> temp =
                boost::const_pointer_cast<cv_bridge::CvImage>(cv_bridge::toCvShare(img, "bgr8"));
            imageProcess(temp);
            findArmor();
            draw();
            target_array_.is_red = target_is_red_;
            target_pub_.publish(target_array_);

        }

        void wide_cam_callback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info)
        {
            if(is_tele_cam_) return;
            ROS_INFO_ONCE("[cam_mode] wide camera mode");
            if (!target_is_armor_)
            {
            //      ROS_INFO("not armor");
            return;
            }
            camera_info_ = info;
            target_array_.header = info->header;
            intrinsics_ = cv::Mat(3, 3, CV_64F, (void*)info->K.data()).clone();
            dist_coeffs_ = cv::Mat(info->D).clone();
            boost::shared_ptr<cv_bridge::CvImage> temp =
                boost::const_pointer_cast<cv_bridge::CvImage>(cv_bridge::toCvShare(img, "bgr8"));
            draw();
            target_array_.is_red = target_is_red_;
            target_pub_.publish(target_array_);

        }



        void fpsCB(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info)
        {
            //todo
        }

         /// HSV
        int red_h_min_low_{};
        int red_h_max_low_{};
        int red_h_min_high_{};
        int red_h_max_high_{};
        int red_s_min_{};
        int red_s_max_{};
        int red_v_min_{};
        int red_v_max_{};
        int blue_h_min_{};
        int blue_h_max_{};
        int blue_s_min_{};
        int blue_s_max_{};
        int blue_v_min_{};
        int blue_v_max_{};

        /// RGB(single channel)
        int binary_thresh_{};

        /// morphology
        int morph_type_{};
        int binary_element_{};

        /// bar morphology
        std::vector<std::vector<cv::Point>> contours_{};

        /// bar compute
        // int bar_br_thresh_{};
        int select_bar_{};
        double max_angle_diff_{};
        float max_lw_ratio_{};
        float min_lw_ratio_{};
        double min_pixel_contained_ratio_{};
        double max_bars_ratio_{};

        /// armor compute
        double min_bars_distance_{};
        double max_bars_distance_{};
        double max_bars_angle_{};
        double max_bars_x_dis_{};
        float warp_white_ratio_{};
        bool select_by_last_{};

        /// armor warp
        double large_armor_ratio_{};
        std::vector<cv::Point2d> warp_reference_;
        bool gamma_{};
        cv::Mat look_up_table_ = cv::Mat::ones(1, 256, CV_8U);
        double contrast_alpha_{};
        double contrast_beta_{};
        double gamma_y_{};
        int warp_thresh_{};
        bool rotate_{};
        /// warp : 32 * 28
        int warp_height_;
        int warp_width_;
        /// cut img (roi)
        int roi_height_{};
        int roi_width_{};
        double top_light_y_{};
        double bottom_light_y_{};
        double bar_length_in_warp_{};
        bool is_large_armor_;
        bool is_large_armor_store_{};
        int armor_temp_[6]{};
        /// id classification
        std::string xml_path_{};
        std::string bin_path_{};
        std::pair<int, float> result_{};
        double negative_confidence_{};
        std::vector<double> expand_ratio_{};
        float id_confidence_{};
        std::vector<int> input_shape_{};
        bool use_id_cls_{};
        float min_id_white_ratio_{};
        float max_id_white_ratio_{};
        double firstnet_score_;
        std::vector<double> softmax_score_;
        ///
        int image_x_center_{};
        int image_y_center_{};

        /// camera coordinate
        std::string camera_coordinate_{};

        /// draw
        DrawImage draw_type_{};
        int line_width_{};

        std::vector<Bar> bars_{};
        std::vector<Armor> armors_{};
        std::vector<Armor> last_frame_armors_{};
        std::vector<Armor> tracked_armors_{};

        std::vector<std::vector<cv::Point2d>> points_{};
        std::vector<int> labels_{};
        std::vector<float> probs_{};
        Object object_{};
        std::mutex obj_locker_;


        cv::Mat raw_image_{};
        cv::Mat binary_image_{};
        cv::Mat morpro_image_{};
        cv::Mat warp_image_{};
        cv::Mat debug_image_{};


        ///debuger
        bool is_bar_debug_{};
        bool is_armor_debug_{};
        int target_option_{};
        bool target_is_red_{};
        int preprocess_method_{};

        //dynamic reconfig
        dynamic_reconfigure::Server<rm_radar_img_proc::PreprocessConfig>* preprocess_cfg_srv_;
        dynamic_reconfigure::Server<rm_radar_img_proc::PreprocessConfig>::CallbackType preprocess_cfg_cb_;
        bool pre_process_dynamic_reconfig_initialized_ = false;

        dynamic_reconfigure::Server<rm_radar_img_proc::ArmorConfig>* armor_cfg_srv_;
        dynamic_reconfigure::Server<rm_radar_img_proc::ArmorConfig>::CallbackType armor_cfg_cb_;
        bool armor_dynamic_reconfig_initialized_ = false;

        dynamic_reconfigure::Server<rm_radar_img_proc::DrawConfig>* draw_cfg_srv_;
        dynamic_reconfigure::Server<rm_radar_img_proc::DrawConfig>::CallbackType draw_cfg_cb_;
        ///

        ///
        rm_radar_msgs::DroneDetectionArray target_array_{};
        ros::Publisher target_pub_{};
        ros::Publisher target_pub_single_{};
         ///
        rm_msgs::TrackData track_data_{};
        ros::Subscriber track_sub_{};
        ros::Publisher track_data_pub_{};

        bool target_is_armor_ = true;

        //solve pnp
        double sz = 0.133 / 2.0; // armor size 130mm x 130mm
        std::vector<cv::Point3d> armor_3d_points_{
            cv::Point3d(-sz, -sz, 0),
            cv::Point3d(-sz, sz, 0),
            cv::Point3d(sz, sz, 0),
            cv::Point3d(sz, -sz, 0)
        };

        void solvePose(const Armor& armor, rm_radar_msgs::DroneDetection& target);
        

        //tf
        std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
        tf2_ros::TransformListener* tf_listener_{nullptr};

    };
}