/*
 * @Author: fwEnjeru666 enjeru2121@gmail.com
 * @Date: 2026-03-28 21:37:12
 * @LastEditors: fwEnjeru666 enjeru2121@gmail.com
 * @LastEditTime: 2026-04-04 16:19:28
 * @FilePath: /rm_ws/src/rm_radarplugin/rm_track/include/motion_models_define.h
 */
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <ros/ros.h>
#include <xmlrpcpp/XmlRpcValue.h>
#include <dynamic_reconfigure/server.h>

namespace rm_radarplugin
{
    struct ModelState
    {
        Eigen::VectorXd state;
        Eigen::MatrixXd covariance;
    };


    class BaseModel
    {
        public:
            virtual ~BaseModel() = default;
            
            virtual ModelState generateInitialState(const Eigen::Vector3d& z_meas) const = 0;
            virtual Eigen::VectorXd processFunction(const Eigen::VectorXd& state, double dt) const = 0;
            virtual Eigen::VectorXd measurementFunction(const Eigen::VectorXd& state, int meas_dim) const = 0;
            
            virtual int getStateAngleIdx() const { return -1; }
            virtual int getMeasurementAngleIdx() const { return -1; }

            virtual Eigen::MatrixXd get_QMatrix() const = 0;
            
            virtual void set_QMatrix(const Eigen::MatrixXd& Q) = 0;

            // Pixel-tight coupling projection params (camera intrinsics + armor size).
            virtual void setProjectionParams(double fx, double fy, double cx, double cy,
                                             double armor_width, double armor_height) {}

            // Projection extrinsic: world/target frame to camera frame.
            virtual void setProjectionExtrinsic(const Eigen::Matrix4d& T_world_to_cam) {}

            //model -> unified model logic
            virtual ModelState toUnifiedState(const ModelState& model_state) const = 0;

            //unified model -> model logic
            virtual ModelState fromUnifiedState(const ModelState& unified_model_state) const = 0;

            void setName(const std::string& name) { model_name_ = name; }
            std::string getName() const { return model_name_; }

        protected:
            std::string model_name_;

            inline void injectDynamicParams(ros::NodeHandle& nh, const XmlRpc::XmlRpcValue& config, const std::vector<std::string>& param_names)
            {
                if (config.hasMember("q_diagonal") && config["q_diagonal"].getType() == XmlRpc::XmlRpcValue::TypeArray)
                {
                    XmlRpc::XmlRpcValue q_arr = const_cast<XmlRpc::XmlRpcValue&>(config)["q_diagonal"];
                    
                    // 遍历传入的名字列表，自动按顺序把数组里的值塞进参数服务器
                    int size = std::min(static_cast<int>(q_arr.size()), static_cast<int>(param_names.size()));
                    for (int i = 0; i < size; ++i)
                    {
                        nh.setParam(param_names[i], static_cast<double>(q_arr[i]));
                    }
                }
            }
    };


    class model_register
    {
        public:
            model_register() = default;
            explicit model_register(const ros::NodeHandle& nh);
            ~model_register() = default;

            int load_models();
            size_t get_model_count() const { return models_.size(); }

            const std::vector<std::shared_ptr<BaseModel>>& get_all_models() const { return models_; }
            const std::vector<std::string>& get_model_names() const { return model_names_; }

            std::shared_ptr<BaseModel> get_model(size_t index)
            {
                return (index < models_.size()) ? models_[index] : nullptr;
            }

        private:
            int load_models_internal();
            std::shared_ptr<BaseModel> create_model_from_config(const XmlRpc::XmlRpcValue& config);
            std::vector<std::shared_ptr<BaseModel>> models_;
            std::vector<std::string> model_names_;

            ros::NodeHandle model_nh_;
            XmlRpc::XmlRpcValue model_configs_;
            bool models_loaded_ = false;

    };

}