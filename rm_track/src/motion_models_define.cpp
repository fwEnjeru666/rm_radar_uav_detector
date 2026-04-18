#include "motion_models_define.h"
#include <rm_track/cvConfig.h>
#include <rm_track/caConfig.h>
#include <rm_track/ctrvConfig.h>
#include <rm_track/singerConfig.h>

namespace rm_radarplugin
{
    namespace
    {
    Eigen::VectorXd projectArmorCorners(const Eigen::VectorXd& state,
                                        double fx, double fy, double cx, double cy,
                                        double armor_width, double armor_height,
                                        double yaw,
                                        const Eigen::Matrix4d& T_world_to_cam)
    {
        Eigen::VectorXd z = Eigen::VectorXd::Zero(8);
        if (state.size() < 3)
        {
            return z;
        }

        const double x = state(0);
        const double y = state(1);
        const double hw = 0.5 * armor_width;
        const double hh = 0.5 * armor_height;
        const double corners[4][2] = {
            {-hw, hh},
            {hw, hh},
            {hw, -hh},
            {-hw, -hh}
        };
        const double c = std::cos(yaw);
        const double s = std::sin(yaw);

        for (int i = 0; i < 4; ++i)
        {
            const double local_x = corners[i][0];
            const double local_y = corners[i][1];
            const double rot_x = c * local_x - s * local_y;
            const double rot_y = s * local_x + c * local_y;
            Eigen::Vector4d pw;
            pw << x + rot_x, y + rot_y, state(2), 1.0;
            Eigen::Vector4d pc = T_world_to_cam * pw;
            const double xc = pc(0);
            const double yc = pc(1);
            const double zc = pc(2);
            if (zc <= 1e-3)
            {
                continue;
            }
            z(2 * i) = fx * (xc / zc) + cx;
            z(2 * i + 1) = fy * (yc / zc) + cy;
        }
        return z;
    }
    } // namespace



//define each models

//模型拓展模板
    // class $model_type$ : public BaseModel
    // {
    // public:
    //     $model_type$() = default;
    //     explicit $model_type$(const XmlRpc::XmlRpcValue& config, ros::NodeHandle& nh) :nh_(nh)
    //     {
    //         dsrv_ = std::make_shared<dynamic_reconfigure::Server<rm_radarplugin::modelConfig>>(nh_);
    //         dynamic_reconfigure::Server<rm_radarplugin::modelConfig>::CallbackType cb = [this](rm_radarplugin::modelConfig& config, uint32_t level)
    //         {
    //             //set Q
    //         };
    //         dsrv_->setCallback(cb);
    //     }
    //     ~$model_type$()  override = default;
    //     virtual generateInitialState(const Eigen::Vector3d& z_meas) const override;
    //     Eigen::VectorXd processFunction(const Eigen::VectorXd& state, double dt) const override{};
    //     Eigen::VectorXd measurementFunction(const Eigen::VectorXd& state, int meas_dim) const override{};
        //    int getStateAngleIdx() const override { return -1; } // 如果模型状态中有角度，返回其索引，否则返回-1
    //     ModelState toUnifiedState(const ModelState& model_state) const override{};
    //     ModelState fromUnifiedState(const ModelState& unified_model_state) const override{};
    //     Eigen::MatrixXd get_QMatrix() const override {return Q_;}
    //     void set_QMatrix(const Eigen::MatrixXd& Q) override { Q_ = Q; }

        
    // private:
    //     Eigen::MatrixXd Q_ = Eigen::MatrixXd::Identity(model_Qdim, model_Qdim) * 0.1;
    //     ros::NodeHandle nh_;
    //     std::shared_ptr<dynamic_reconfigure::Server<rm_radarplugin::modelConfig>> dsrv_;
    // };


    //CV模型状态: [x, y, z, vx, vy, vz]
    class CV : public BaseModel
    {
    public:
        CV() = default;
        explicit CV(const XmlRpc::XmlRpcValue& config, ros::NodeHandle& nh) :nh_(nh)
        {
            injectDynamicParams(nh_, config, {"q_x", "q_y", "q_z", "q_vx", "q_vy", "q_vz"});
            dsrv_ = std::make_shared<dynamic_reconfigure::Server<rm_radarplugin::cvConfig>>(nh_);
            dynamic_reconfigure::Server<rm_radarplugin::cvConfig>::CallbackType cb = [this](rm_radarplugin::cvConfig& config, uint32_t level)
            {
                //set Q
                Q_(0, 0) = config.q_x;
                Q_(1, 1) = config.q_y;
                Q_(2, 2) = config.q_z;
                Q_(3, 3) = config.q_vx;
                Q_(4, 4) = config.q_vy;
                Q_(5, 5) = config.q_vz;
            };
            dsrv_->setCallback(cb);
        }

        ~CV() override = default;

        ModelState generateInitialState(const Eigen::Vector3d& z_meas) const override
        {
            int z_dim = z_meas.size(); 
                
            ModelState init_state;
            init_state.state = Eigen::VectorXd::Zero(6);
            init_state.state.head(z_dim) = z_meas; 
            
            init_state.covariance = Eigen::MatrixXd::Identity(6, 6) * 10.0;            

            init_state.covariance.topLeftCorner(z_dim, z_dim) = Eigen::MatrixXd::Identity(z_dim, z_dim) * 0.1;
            
            return init_state;
        }

        Eigen::VectorXd processFunction(const Eigen::VectorXd& state, double dt) const override
        {
            // CV模型状态: [x, y, z, vx, vy, vz]
            Eigen::VectorXd next_state = state;
            next_state(0) += state(3) * dt;
            next_state(1) += state(4) * dt;
            next_state(2) += state(5) * dt;
            return next_state;
        }

        Eigen::VectorXd measurementFunction(const Eigen::VectorXd& state, int meas_dim) const override
        {
            if (meas_dim == 8)
            {
                double yaw = 0.0;
                if (state.size() >= 5)
                {
                    const double vx = state(3);
                    const double vy = state(4);
                    if (std::hypot(vx, vy) > 1e-3)
                    {
                        yaw = std::atan2(vy, vx);
                    }
                }
                return projectArmorCorners(state, fx_, fy_, cx_, cy_, armor_width_, armor_height_, yaw, T_world_to_cam_);
            }
            return state.head(meas_dim);
        }

        void setProjectionParams(double fx, double fy, double cx, double cy,
                                 double armor_width, double armor_height) override
        {
            fx_ = fx;
            fy_ = fy;
            cx_ = cx;
            cy_ = cy;
            armor_width_ = armor_width;
            armor_height_ = armor_height;
        }

        void setProjectionExtrinsic(const Eigen::Matrix4d& T_world_to_cam) override
        {
            T_world_to_cam_ = T_world_to_cam;
        }

        int getStateAngleIdx() const override { return -1; } // CV模型状态中没有角度，返回-1

        Eigen::MatrixXd get_QMatrix() const override {return Q_;} 

        ModelState toUnifiedState(const ModelState& model_state) const override
        {
            //cv unified State维度为9: [x, y, z, vx, vy, vz, ax, ay, az]
            ModelState unified;
            unified.state = Eigen::VectorXd::Zero(9);
            unified.covariance = Eigen::MatrixXd::Zero(9, 9);

            unified.state.head(6) = model_state.state;
            unified.covariance.topLeftCorner(6, 6) = model_state.covariance;

            return unified;
        }

        ModelState fromUnifiedState(const ModelState& unified_model_state) const override
        {
            ModelState cv_state;
            // 从 9D 映射回 6D cv
            cv_state.state = unified_model_state.state.head(6);
            cv_state.covariance = unified_model_state.covariance.topLeftCorner(6, 6);
            
            return cv_state;
        }

        void set_QMatrix(const Eigen::MatrixXd& Q) override { Q_ = Q; }

            
    private:
        Eigen::MatrixXd Q_ = Eigen::MatrixXd::Identity(6, 6) * 0.1;
        ros::NodeHandle nh_;
        std::shared_ptr<dynamic_reconfigure::Server<rm_radarplugin::cvConfig>> dsrv_;
        double fx_{1.0};
        double fy_{1.0};
        double cx_{0.0};
        double cy_{0.0};
        double armor_width_{0.135};
        double armor_height_{0.055};
        Eigen::Matrix4d T_world_to_cam_ = Eigen::Matrix4d::Identity();
    };

    //CA模型状态: [x, y, z, vx, vy, vz, ax, ay, az]
    class CA : public BaseModel
    {
        public:
            CA() = default;
            explicit CA(const XmlRpc::XmlRpcValue& config, ros::NodeHandle& nh) :nh_(nh)
            {
                injectDynamicParams(nh_, config, {"q_x", "q_y", "q_z", "q_vx", "q_vy", "q_vz", "q_ax", "q_ay", "q_az"});
                dsrv_ = std::make_shared<dynamic_reconfigure::Server<rm_radarplugin::caConfig>>(nh_);
                dynamic_reconfigure::Server<rm_radarplugin::caConfig>::CallbackType cb = [this](rm_radarplugin::caConfig& config, uint32_t level)
                {
                    //set Q
                    this->Q_.setZero();
                    this->Q_(0, 0) = config.q_x;
                    this->Q_(1, 1) = config.q_y;
                    this->Q_(2, 2) = config.q_z;
                    this->Q_(3, 3) = config.q_vx;
                    this->Q_(4, 4) = config.q_vy;
                    this->Q_(5, 5) = config.q_vz;
                    this->Q_(6, 6) = config.q_ax;
                    this->Q_(7, 7) = config.q_ay;
                    this->Q_(8, 8) = config.q_az;
                };
                dsrv_->setCallback(cb);
            }

            ~CA() override = default;

            ModelState generateInitialState(const Eigen::Vector3d& z_meas) const override
            {
                int z_dim = z_meas.size();
    
                ModelState init_state;
                init_state.state = Eigen::VectorXd::Zero(9);
                init_state.state.head(z_dim) = z_meas;
                
                init_state.covariance = Eigen::MatrixXd::Identity(9, 9) * 10.0; // 默认全未知
                init_state.covariance.topLeftCorner(z_dim, z_dim) = Eigen::MatrixXd::Identity(z_dim, z_dim) * 0.1; // 覆盖测到的维度
                
                return init_state;
            }
            Eigen::VectorXd processFunction(const Eigen::VectorXd& state, double dt) const override
            {
                Eigen::VectorXd next_state = state;
                double t2 = 0.5 * dt * dt;
                next_state(0) += state(3) * dt + state(6) * t2; // x + vx*dt + 0.5*ax*dt^2
                next_state(1) += state(4) * dt + state(7) * t2; // y + vy*dt + 0.5*ay*dt^2
                next_state(2) += state(5) * dt + state(8) * t2; // z + vz*dt + 0.5*az*dt^2
                next_state(3) += state(6) * dt; // vx + ax*dt
                next_state(4) += state(7) * dt; // vy + ay
                next_state(5) += state(8) * dt; // vz + az*dt
                return next_state;
            }

            Eigen::VectorXd measurementFunction(const Eigen::VectorXd& state, int meas_dim) const override
            {
                if (meas_dim == 8)
                {
                    double yaw = 0.0;
                    if (state.size() >= 5)
                    {
                        const double vx = state(3);
                        const double vy = state(4);
                        if (std::hypot(vx, vy) > 1e-3)
                        {
                            yaw = std::atan2(vy, vx);
                        }
                    }
                    return projectArmorCorners(state, fx_, fy_, cx_, cy_, armor_width_, armor_height_, yaw, T_world_to_cam_);
                }
                return state.head(meas_dim);
            }

            void setProjectionParams(double fx, double fy, double cx, double cy,
                                     double armor_width, double armor_height) override
            {
                fx_ = fx;
                fy_ = fy;
                cx_ = cx;
                cy_ = cy;
                armor_width_ = armor_width;
                armor_height_ = armor_height;
            }

            void setProjectionExtrinsic(const Eigen::Matrix4d& T_world_to_cam) override
            {
                T_world_to_cam_ = T_world_to_cam;
            }

            int getStateAngleIdx() const override { return -1; } // No angle state in CA model
            ModelState toUnifiedState(const ModelState& model_state) const override
            {
                return model_state; // CA state already matches unified state
            }

            ModelState fromUnifiedState(const ModelState& unified_model_state) const override
            {
                return unified_model_state; // CA state already matches unified state
            }
            Eigen::MatrixXd get_QMatrix() const override {return Q_;}
            void set_QMatrix(const Eigen::MatrixXd& Q) override { Q_ = Q;}

        private:
            Eigen::MatrixXd Q_ = Eigen::MatrixXd::Identity(9, 9) * 0.1;
            ros::NodeHandle nh_;
            std::shared_ptr<dynamic_reconfigure::Server<rm_radarplugin::caConfig>> dsrv_;
            double fx_{1.0};
            double fy_{1.0};
            double cx_{0.0};
            double cy_{0.0};
            double armor_width_{0.135};
            double armor_height_{0.055};
            Eigen::Matrix4d T_world_to_cam_ = Eigen::Matrix4d::Identity();
    };


    //CTRV模型状态: [x, y, z, v, yaw, yaw_rate, vz]
    class CTRV : public BaseModel
    {
        public:
            CTRV() = default;
            explicit CTRV(const XmlRpc::XmlRpcValue& config, ros::NodeHandle& nh) :nh_(nh)
            {
                injectDynamicParams(nh_, config, {"q_x", "q_y", "q_z", "q_v", "q_vz","q_yaw", "q_yaw_rate", });
                dsrv_ = std::make_shared<dynamic_reconfigure::Server<rm_radarplugin::ctrvConfig>>(nh_);
                dynamic_reconfigure::Server<rm_radarplugin::ctrvConfig>::CallbackType cb = [this](rm_radarplugin::ctrvConfig& config, uint32_t level)
                {
                    this->Q_.setZero();
                    this->Q_(0, 0) = config.q_x;
                    this->Q_(1, 1) = config.q_y;
                    this->Q_(2, 2) = config.q_z;
                    this->Q_(3, 3) = config.q_v;          // 索引 3 是平面合速度 v
                    this->Q_(4, 4) = config.q_yaw;        // 索引 4 是偏航角 yaw
                    this->Q_(5, 5) = config.q_yaw_rate;   // 索引 5 是角速度 w
                    this->Q_(6, 6) = config.q_vz;         // 索引 6 是 Z 轴速度 vz
                };
                dsrv_->setCallback(cb);
            }

            ~CTRV() override = default;
            ModelState generateInitialState(const Eigen::Vector3d& z_meas) const override
            {
                int z_dim = z_meas.size();
                    
                ModelState init_state;
                init_state.state = Eigen::VectorXd::Zero(7);
                init_state.state.head(z_dim) = z_meas;
                
                init_state.covariance = Eigen::MatrixXd::Identity(7, 7) * 10.0; // 默认全未知
                
                // 覆盖测量的维度
                init_state.covariance.topLeftCorner(z_dim, z_dim) = Eigen::MatrixXd::Identity(z_dim, z_dim) * 0.1;
                
                //把角度的方差强行压下来，防止爆炸
                init_state.covariance(4, 4) = 0.5;  // yaw
                init_state.covariance(5, 5) = 0.1;  // yaw_rate
                
                return init_state;
            }

            Eigen::VectorXd processFunction(const Eigen::VectorXd& state, double dt) const override
            {
                Eigen::VectorXd next_state = state;
                double v = state(3) , yaw = state(4), yaw_rate = state(5) , vz = state(6);
                //根据CTRV模型的运动方程进行状态预测
                if(std::abs(yaw_rate) > 1e-5)
                {
                    next_state(0) += (v / yaw_rate) * (std::sin(yaw + yaw_rate * dt) - std::sin(yaw));
                    next_state(1) += (v / yaw_rate) * (-std::cos(yaw + yaw_rate * dt) + std::cos(yaw));
                }
                else
                {
                    next_state(0) += v * std::cos(yaw) * dt;
                    next_state(1) += v * std::sin(yaw) * dt;
                }
                next_state(2) += vz * dt; // z 方向匀速
                next_state(4) += yaw_rate * dt; // 更新航向角
                return next_state;
            }

            Eigen::VectorXd measurementFunction(const Eigen::VectorXd& state, int meas_dim) const override
            {
                if (meas_dim == 8)
                {
                    const double yaw = (state.size() > 4) ? state(4) : 0.0;
                    return projectArmorCorners(state, fx_, fy_, cx_, cy_, armor_width_, armor_height_, yaw, T_world_to_cam_);
                }
                return state.head(meas_dim);
            }

            void setProjectionParams(double fx, double fy, double cx, double cy,
                                     double armor_width, double armor_height) override
            {
                fx_ = fx;
                fy_ = fy;
                cx_ = cx;
                cy_ = cy;
                armor_width_ = armor_width;
                armor_height_ = armor_height;
            }

            void setProjectionExtrinsic(const Eigen::Matrix4d& T_world_to_cam) override
            {
                T_world_to_cam_ = T_world_to_cam;
            }

            int getStateAngleIdx() const override { return 4; } // yaw 在状态向量中的索引

            ModelState toUnifiedState(const ModelState& model_state) const override
            {
                ModelState unified;
                unified.state = Eigen::VectorXd::Zero(9);
                double v = model_state.state(3) , yaw = model_state.state(4), yaw_rate = model_state.state(5) , vz = model_state.state(6);
                unified.state(0) = model_state.state(0); // x
                unified.state(1) = model_state.state(1); // y
                unified.state(2) = model_state.state(2); // z
                unified.state(3) = v*std::cos(yaw); // vx
                unified.state(4) = v*std::sin(yaw); // vy
                unified.state(5) = vz; // vz
                unified.state(6) = -v*yaw_rate*std::sin(yaw); // ax
                unified.state(7) = v*yaw_rate*std::cos(yaw); // ay
                unified.state(8) = 0.0; // az = 0

                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(9, 7); // 9D unified state 对 7D CTRV state 的雅可比矩阵
                J(0, 0) = 1.0; // x - x 
                J(1, 1) = 1.0; // y - y 
                J(2, 2) = 1.0; // z - z 
                J(3, 3) = std::cos(yaw); // vx - v 
                J(3, 4) = -v * std::sin(yaw); // vx - yaw 
                J(3, 5) = 0.0; // vx - yaw_rate 
                J(4, 3) = std::sin(yaw); // vy - v 
                J(4, 4) = v * std::cos(yaw); // vy - yaw
                J(4, 5) = 0.0; // vy - yaw_rate
                J(5, 6) = 1.0; // vz - vz 
                J(6, 3) = -yaw_rate * std::sin(yaw); // ax - v 
                J(6, 4) = -v * yaw_rate * std::cos(yaw); // ax - yaw 
                J(6, 5) = -v * std::sin(yaw); // ax - yaw_rate 
                J(7, 3) = yaw_rate * std::cos(yaw); // ay - v 
                J(7, 4) = -v * yaw_rate * std::sin(yaw); // ay - yaw 
                J(7, 5) = v * std::cos(yaw); // ay - yaw_rate 
                // az 对任何状态变量的偏导数都是0

                unified.covariance = J * model_state.covariance * J.transpose(); // 通过雅可比矩阵进行协方差转换
                return unified;
            }
            ModelState fromUnifiedState(const ModelState& unified_model_state) const override
            {
                ModelState ctrv_state;
                // 从 9D 映射回 7D CTRV
                double vx = unified_model_state.state(3);
                double vy = unified_model_state.state(4);
                double v = std::sqrt(vx*vx + vy*vy);
                double yaw = std::atan2(vy, vx);
                double vz = unified_model_state.state(5);

                ctrv_state.state = Eigen::VectorXd::Zero(7);
                ctrv_state.state(0) = unified_model_state.state(0); // x
                ctrv_state.state(1) = unified_model_state.state(1); // y
                ctrv_state.state(2) = unified_model_state.state(2); // z
                ctrv_state.state(3) = v; // v
                ctrv_state.state(4) = yaw; // yaw
                ctrv_state.state(5) = 0.0; // yaw_rate 由于统一状态中没有yaw_rate信息，这里暂时设置为0，实际使用中可能需要根据具体情况进行估计或设置
                ctrv_state.state(6) = vz; // vz

                Eigen::MatrixXd J_inv = Eigen::MatrixXd::Zero(7, 9); // 7D CTRV state 对 9D unified state 的逆雅可比矩阵
                J_inv(0, 0) = 1.0; // x - x 
                J_inv(1, 1) = 1.0; // y - y 
                J_inv(2, 2) = 1.0; // z - z
                
                //除0保护
                double v2 = vx*vx + vy*vy;
                if(v2 > 1e-5)
                {
                    J_inv(3, 3) = vx / v; // v - vx 
                    J_inv(3, 4) = vy / v; // v - vy 
                    J_inv(4, 3) = -vy / (v*v); // yaw - vx 
                    J_inv(4, 4) = vx / (v*v); // yaw - vy 
                }
                else
                {
                    J_inv(3, 3) = 0.0; 
                    J_inv(3, 4) = 0.0;
                    J_inv(4, 3) = 0.0;
                    J_inv(4, 4) = 0.0;
                }
                J_inv(6, 5) = 1.0; // vz - vz 

                ctrv_state.covariance = J_inv * unified_model_state.covariance * J_inv.transpose(); // 通过逆雅可比矩阵进行协方差转换
                ctrv_state.covariance(5, 5) += 0.1; // 由于yaw_rate在fromUnifiedState中被设置为0，这里增加一些不确定性
                return ctrv_state;
            }
            Eigen::MatrixXd get_QMatrix() const override {return Q_;}
            void set_QMatrix(const Eigen::MatrixXd& Q) override {Q_ = Q;}

        private:
            Eigen::MatrixXd Q_ = Eigen::MatrixXd::Identity(7, 7) * 0.1;
            ros::NodeHandle nh_;
            std::shared_ptr<dynamic_reconfigure::Server<rm_radarplugin::ctrvConfig>> dsrv_;
            double fx_{1.0};
            double fy_{1.0};
            double cx_{0.0};
            double cy_{0.0};
            double armor_width_{0.135};
            double armor_height_{0.055};
            Eigen::Matrix4d T_world_to_cam_ = Eigen::Matrix4d::Identity();
    };

    // Singer模型状态: [x, y, z, vx, vy, vz, ax, ay, az]
    // 加速度服从一阶马尔可夫衰减: a(k+1) = exp(-dt/tau) * a(k)
    class Singer : public BaseModel
    {
        public:
            Singer() = default;
            explicit Singer(const XmlRpc::XmlRpcValue& config, ros::NodeHandle& nh) : nh_(nh)
            {
                injectDynamicParams(nh_, config, {"q_x", "q_y", "q_z", "q_vx", "q_vy", "q_vz", "q_ax", "q_ay", "q_az"});

                if (config.hasMember("tau"))
                {
                    tau_ = std::max(0.05, static_cast<double>(config["tau"]));
                }

                dsrv_ = std::make_shared<dynamic_reconfigure::Server<rm_radarplugin::singerConfig>>(nh_);
                dynamic_reconfigure::Server<rm_radarplugin::singerConfig>::CallbackType cb = [this](rm_radarplugin::singerConfig& config, uint32_t level)
                {
                    (void)level;
                    this->Q_.setZero();
                    this->Q_(0, 0) = config.q_x;
                    this->Q_(1, 1) = config.q_y;
                    this->Q_(2, 2) = config.q_z;
                    this->Q_(3, 3) = config.q_vx;
                    this->Q_(4, 4) = config.q_vy;
                    this->Q_(5, 5) = config.q_vz;
                    this->Q_(6, 6) = config.q_ax;
                    this->Q_(7, 7) = config.q_ay;
                    this->Q_(8, 8) = config.q_az;
                    this->tau_ = std::max(0.05, config.tau);
                };
                dsrv_->setCallback(cb);
            }

            ~Singer() override = default;

            ModelState generateInitialState(const Eigen::Vector3d& z_meas) const override
            {
                int z_dim = z_meas.size();

                ModelState init_state;
                init_state.state = Eigen::VectorXd::Zero(9);
                init_state.state.head(z_dim) = z_meas;

                init_state.covariance = Eigen::MatrixXd::Identity(9, 9) * 10.0;
                init_state.covariance.topLeftCorner(z_dim, z_dim) = Eigen::MatrixXd::Identity(z_dim, z_dim) * 0.1;

                return init_state;
            }

            Eigen::VectorXd processFunction(const Eigen::VectorXd& state, double dt) const override
            {
                Eigen::VectorXd next_state = state;
                const double t2 = 0.5 * dt * dt;
                const double decay = std::exp(-dt / std::max(0.05, tau_));

                next_state(0) += state(3) * dt + state(6) * t2;
                next_state(1) += state(4) * dt + state(7) * t2;
                next_state(2) += state(5) * dt + state(8) * t2;
                next_state(3) += state(6) * dt;
                next_state(4) += state(7) * dt;
                next_state(5) += state(8) * dt;

                next_state(6) = state(6) * decay;
                next_state(7) = state(7) * decay;
                next_state(8) = state(8) * decay;

                return next_state;
            }

            Eigen::VectorXd measurementFunction(const Eigen::VectorXd& state, int meas_dim) const override
            {
                if (meas_dim == 8)
                {
                    double yaw = 0.0;
                    if (state.size() >= 5)
                    {
                        const double vx = state(3);
                        const double vy = state(4);
                        if (std::hypot(vx, vy) > 1e-3)
                        {
                            yaw = std::atan2(vy, vx);
                        }
                    }
                    return projectArmorCorners(state, fx_, fy_, cx_, cy_, armor_width_, armor_height_, yaw, T_world_to_cam_);
                }
                return state.head(meas_dim);
            }

            void setProjectionParams(double fx, double fy, double cx, double cy,
                                     double armor_width, double armor_height) override
            {
                fx_ = fx;
                fy_ = fy;
                cx_ = cx;
                cy_ = cy;
                armor_width_ = armor_width;
                armor_height_ = armor_height;
            }

            void setProjectionExtrinsic(const Eigen::Matrix4d& T_world_to_cam) override
            {
                T_world_to_cam_ = T_world_to_cam;
            }

            int getStateAngleIdx() const override { return -1; }

            ModelState toUnifiedState(const ModelState& model_state) const override
            {
                return model_state;
            }

            ModelState fromUnifiedState(const ModelState& unified_model_state) const override
            {
                return unified_model_state;
            }

            Eigen::MatrixXd get_QMatrix() const override { return Q_; }
            void set_QMatrix(const Eigen::MatrixXd& Q) override { Q_ = Q; }

        private:
            Eigen::MatrixXd Q_ = Eigen::MatrixXd::Identity(9, 9) * 0.1;
            ros::NodeHandle nh_;
            std::shared_ptr<dynamic_reconfigure::Server<rm_radarplugin::singerConfig>> dsrv_;
            double tau_{0.8};
            double fx_{1.0};
            double fy_{1.0};
            double cx_{0.0};
            double cy_{0.0};
            double armor_width_{0.135};
            double armor_height_{0.055};
            Eigen::Matrix4d T_world_to_cam_ = Eigen::Matrix4d::Identity();
    };

    //end define each models

    
    //define model register
    model_register::model_register(const ros::NodeHandle& nh) : model_nh_(nh)
    {
        if (!model_nh_.getParam("models", model_configs_)) 
        {
        ROS_ERROR_STREAM("[" << model_nh_.getNamespace() << "] No 'models' found on parameter server.");
        }
    }

    int model_register::load_models()
    {
        if (models_loaded_)
        {
            return models_.size();
        }
        return load_models_internal();
    }

    int model_register::load_models_internal()
    {
        if (!model_nh_.getParam("models", model_configs_))
        {
            ROS_ERROR("No 'models' config found on parameter server.");
            return 0;
        }

        if (model_configs_.getType() != XmlRpc::XmlRpcValue::TypeArray)
        {
            ROS_ERROR("'models' config is not an array.");
            return 0;
        }

        for (int i = 0; i < model_configs_.size(); ++i)
        {
            if (model_configs_[i].getType() != XmlRpc::XmlRpcValue::TypeStruct)
            {
                ROS_WARN("Model config at index %d is not a struct.", i);
                continue;
            }

            std::shared_ptr<BaseModel> model = create_model_from_config(model_configs_[i]);
            if (model)
            {
                models_.push_back(model);
                model_names_.push_back(model->getName());
            }
        }

        models_loaded_ = !models_.empty();
        return models_.size();
    }


    //如果要新增一个model 只需要在这里添加一个分支，并实现对应的类即可
    std::shared_ptr<BaseModel> model_register::create_model_from_config(const XmlRpc::XmlRpcValue& config)
    {
        std::shared_ptr<BaseModel> model = nullptr;
        //读取cfg内的type字段
        if (config.getType() != XmlRpc::XmlRpcValue::TypeStruct)
        {
            ROS_ERROR_STREAM("[" << model_nh_.getNamespace() << "] Model config is not a struct.");
            return nullptr;
        }
        if (!config.hasMember("type"))
        {
            ROS_ERROR_STREAM("[" << model_nh_.getNamespace() << "] Model config missing 'type' field when loading model.");
            return nullptr;
        }

        std::string type = static_cast<std::string>(config["type"]);
        std::string name = config.hasMember("name") ? static_cast<std::string>(config["name"]) : type;

        ros::NodeHandle sub_nh(model_nh_, "models/" + name);

        if (type == "CV")
        {
            model = std::make_shared<CV>(config, sub_nh);
        }
        else if (type == "CA")
        {
            model = std::make_shared<CA>(config, sub_nh);
        }
        else if (type == "CTRV")
        {
            model = std::make_shared<CTRV>(config, sub_nh);
        }
        else if (type == "Singer")
        {
            model = std::make_shared<Singer>(config, sub_nh);
        }
        
        if(model)
        {
            model->setName(name);
            return model;
        }

        ROS_FATAL_STREAM("[" << model_nh_.getNamespace() << "] Unknown model type: " << type);

        return nullptr;
    }



//end define model register


}