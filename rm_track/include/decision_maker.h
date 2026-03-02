#pragma once

#include <ros/ros.h>
#include <rm_radar_msgs/DecisionState.h>
#include "common.h"
#include <ros/ros.h>

namespace rm_radarplugin
{
    struct DeviceState
    {
        bool radar_found;
        bool tele_found;
        bool wide_found;
    };


    class DecisionMaker
    {
    public:
        DecisionMaker();
        ~DecisionMaker() = default;

        void updateDecisionState(DeviceState& device_state);
        rm_track::DecisionState getCurrentDecisionState() const {return current_decision_state_;};
        

    private:
        rm_track::DecisionState current_decision_state_;
        DeviceState current_device_state_;
        ros::Time last_update_time_;
        int time_limits_ = 2; // seconds
    };
} // namespace rm_radarplugin