/**
 * @file imgproc_test_node.cpp
 * @brief Standalone test node for rm_radar_img_proc Processor.
 *
 * Feeds a single image or video file into the same camera topic that
 * Processor subscribes to, so you can test the full imgproc pipeline
 * (preprocess → binary → morphology → findBars → findArmor → draw)
 * without hardware cameras or rosbag.
 *
 * Usage:
 *   roslaunch rm_radar_img_proc test_imgproc.launch image:=/path/to/image.jpg
 *   roslaunch rm_radar_img_proc test_imgproc.launch video:=/path/to/video.mp4
 *   roslaunch rm_radar_img_proc test_imgproc.launch image:=/path/to/image.jpg camera_info:=/path/to/calib.yaml
 *
 * Keyboard controls (when OpenCV window is focused):
 *   [Space]  - pause / resume video playback
 *   [S]      - step one frame (while paused)
 *   [R]      - restart from beginning
 *   [Q/Esc]  - quit
 *   [+/-]    - speed up / slow down playback
 */

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "imgproc_test_node");
    ros::NodeHandle nh("~");

    // =================== Parameters ===================
    std::string image_path, video_path, camera_info_url;
    nh.param<std::string>("image", image_path, "");
    nh.param<std::string>("video", video_path, "");
    nh.param<std::string>("camera_info_url", camera_info_url, "");
    
    // Publish topic (must match Processor's subscription)
    std::string image_topic;
    nh.param<std::string>("image_topic", image_topic, "/hk_camera/image_raw");
    
    double fps;
    nh.param("fps", fps, 30.0);
    
    bool loop;
    nh.param("loop", loop, true);

    if (image_path.empty() && video_path.empty())
    {
        ROS_FATAL("[TestNode] Must specify either ~image:=/path/to/img or ~video:=/path/to/video");
        return 1;
    }

    // =================== Camera Info ===================
    // Build a default CameraInfo (identity intrinsics, no distortion).
    // Overridden if camera_info_url is provided.
    sensor_msgs::CameraInfo cam_info;
    std::unique_ptr<camera_info_manager::CameraInfoManager> cinfo_mgr;
    
    if (!camera_info_url.empty())
    {
        cinfo_mgr = std::make_unique<camera_info_manager::CameraInfoManager>(nh, "test_camera", camera_info_url);
        if (cinfo_mgr->isCalibrated())
        {
            cam_info = cinfo_mgr->getCameraInfo();
            ROS_INFO("[TestNode] Loaded calibration from: %s", camera_info_url.c_str());
        }
        else
        {
            ROS_WARN("[TestNode] camera_info_url provided but not calibrated, using defaults");
        }
    }

    // =================== Image Transport Publisher ===================
    image_transport::ImageTransport it(nh);
    image_transport::CameraPublisher cam_pub = it.advertiseCamera(image_topic, 1);
    
    ROS_INFO("[TestNode] Publishing to: %s", image_topic.c_str());

    // =================== Load media ===================
    cv::VideoCapture cap;
    cv::Mat single_image;
    bool is_video = false;

    if (!video_path.empty())
    {
        cap.open(video_path);
        if (!cap.isOpened())
        {
            ROS_FATAL("[TestNode] Cannot open video: %s", video_path.c_str());
            return 1;
        }
        is_video = true;
        // Override fps from video if available
        double video_fps = cap.get(cv::CAP_PROP_FPS);
        if (video_fps > 0) fps = video_fps;
        ROS_INFO("[TestNode] Video: %s (%.1f fps, %d frames)",
                 video_path.c_str(), fps,
                 (int)cap.get(cv::CAP_PROP_FRAME_COUNT));
    }
    else
    {
        single_image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (single_image.empty())
        {
            ROS_FATAL("[TestNode] Cannot read image: %s", image_path.c_str());
            return 1;
        }
        ROS_INFO("[TestNode] Image: %s (%dx%d)", 
                 image_path.c_str(), single_image.cols, single_image.rows);
    }

    // =================== Build default CameraInfo if not loaded ===================
    auto setupDefaultCamInfo = [&](int w, int h) {
        if (cam_info.width == 0)
        {
            cam_info.width = w;
            cam_info.height = h;
            cam_info.distortion_model = "plumb_bob";
            cam_info.D = {0, 0, 0, 0, 0};
            // Identity intrinsics — approximate focal length = image width
            double fx = w;
            double fy = w;
            double cx = w / 2.0;
            double cy = h / 2.0;
            cam_info.K = {fx, 0, cx,  0, fy, cy,  0, 0, 1};
            cam_info.R = {1, 0, 0,  0, 1, 0,  0, 0, 1};
            cam_info.P = {fx, 0, cx, 0,  0, fy, cy, 0,  0, 0, 1, 0};
            ROS_WARN("[TestNode] Using default identity intrinsics (no calibration loaded)");
        }
    };

    // =================== Playback controls ===================
    bool paused = false;
    bool step_one = false;
    int delay_ms = std::max(1, (int)(1000.0 / fps));
    
    // Create a small control window
    const std::string win_name = "ImgProc Test [Space=pause, S=step, R=restart, Q=quit]";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    
    ROS_INFO("[TestNode] Controls: Space=pause, S=step, R=restart, Q/Esc=quit, +/-=speed");
    ROS_INFO("[TestNode] Starting playback at %.1f fps (delay=%dms)", fps, delay_ms);

    ros::Rate rate(fps);
    int frame_idx = 0;

    while (ros::ok())
    {
        cv::Mat frame;

        if (is_video)
        {
            if (!paused || step_one)
            {
                if (!cap.read(frame))
                {
                    if (loop)
                    {
                        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                        frame_idx = 0;
                        ROS_INFO("[TestNode] Video looped.");
                        continue;
                    }
                    else
                    {
                        ROS_INFO("[TestNode] Video ended.");
                        break;
                    }
                }
                frame_idx++;
                step_one = false;
            }
            else
            {
                // Paused — just show last frame
                // We need to keep the event loop alive for key handling
                int key = cv::waitKey(50) & 0xFF;
                if (key == ' ') paused = false;
                else if (key == 's' || key == 'S') step_one = true;
                else if (key == 'r' || key == 'R') { cap.set(cv::CAP_PROP_POS_FRAMES, 0); frame_idx = 0; paused = false; }
                else if (key == 'q' || key == 'Q' || key == 27) break;
                ros::spinOnce();
                continue;
            }
        }
        else
        {
            // Single image mode — republish the same image every cycle
            frame = single_image.clone();
        }

        if (frame.empty()) continue;

        // Ensure CameraInfo dimensions match
        setupDefaultCamInfo(frame.cols, frame.rows);

        // Build ROS message
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        header.frame_id = "camera_optical_frame";

        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
        cam_info.header = header;

        // Publish
        cam_pub.publish(*img_msg, cam_info);

        // Show frame locally with info overlay
        cv::Mat display = frame.clone();
        std::string info_text;
        if (is_video)
        {
            int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
            char buf[128];
            snprintf(buf, sizeof(buf), "Frame %d/%d | %.1f fps%s", 
                     frame_idx, total_frames, fps, paused ? " [PAUSED]" : "");
            info_text = buf;
        }
        else
        {
            info_text = "Single Image Mode (re-publishing)";
        }
        cv::putText(display, info_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::imshow(win_name, display);

        // Handle keyboard
        int key = cv::waitKey(delay_ms) & 0xFF;
        if (key == ' ') paused = !paused;
        else if (key == 's' || key == 'S') { paused = true; step_one = true; }
        else if (key == 'r' || key == 'R') { 
            if (is_video) { cap.set(cv::CAP_PROP_POS_FRAMES, 0); frame_idx = 0; }
            paused = false; 
        }
        else if (key == 'q' || key == 'Q' || key == 27) break;
        else if (key == '+' || key == '=') { fps = std::min(fps * 1.5, 120.0); delay_ms = std::max(1, (int)(1000.0/fps)); ROS_INFO("[TestNode] Speed: %.1f fps", fps); }
        else if (key == '-' || key == '_') { fps = std::max(fps / 1.5, 1.0); delay_ms = std::max(1, (int)(1000.0/fps)); ROS_INFO("[TestNode] Speed: %.1f fps", fps); }

        ros::spinOnce();
        rate.sleep();
    }

    cv::destroyAllWindows();
    ROS_INFO("[TestNode] Finished.");
    return 0;
}
