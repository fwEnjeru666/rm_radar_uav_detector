#include "decision_maker.h"

namespace rm_radarplugin
{       
    DecisionMaker::DecisionMaker()
    {
        current_decision_state_ = rm_track::BLIND_SCAN;
        current_device_state_.radar_found = false;
        current_device_state_.tele_found = false;
        current_device_state_.wide_found = false;
        last_update_time_ = ros::Time::now();
    }


    void DecisionMaker::updateDecisionState(DeviceState& device_state)
    {
        ros::Time now = ros::Time::now();
        double time_in_state = (now - last_update_time_).toSec();

        auto next_state = current_decision_state_;

        switch (current_decision_state_)
        {
        // --------------------------------------------------------
        // 状态 4: 盲搜 (BLIND_SCAN)
        // --------------------------------------------------------
        case rm_track::BLIND_SCAN:
         
            if (device_state.tele_found) {
                next_state = rm_track::TRACK_TELE;
                ROS_INFO("[FSM] Blind -> Tele");
            }
            else if (device_state.wide_found) {
                next_state = rm_track::TRACK_WIDE;
                ROS_INFO("[FSM] Blind -> Wide");
            }
            else if (device_state.radar_found) {
                next_state = rm_track::TRACK_RADAR;
                ROS_INFO("[FSM] Blind -> Radar");
            }
            break;

        // --------------------------------------------------------
        // 状态 2: 雷达引导 (TRACK_RADAR)
        // --------------------------------------------------------
        case rm_track::TRACK_RADAR:
         
            if (device_state.tele_found) {
                next_state = rm_track::TRACK_TELE;
                ROS_INFO("[FSM] Radar -> Tele");
            }
            else if (device_state.wide_found) {
                next_state = rm_track::TRACK_WIDE;
                ROS_INFO("[FSM] Radar -> Wide");
            }
            else if (time_in_state > time_limits_) {
              
                next_state = rm_track::BLIND_SCAN;
                ROS_WARN("[FSM] Radar Timeout -> Blind");
            }
            else if (!device_state.radar_found) {
             
                next_state = rm_track::BLIND_SCAN;
                ROS_WARN("[FSM] Radar Lost -> Blind");
            }
            break;

        // --------------------------------------------------------
        // 状态 1: 短焦跟踪 (TRACK_WIDE)
        // --------------------------------------------------------
        case rm_track::TRACK_WIDE:
            if (device_state.tele_found) {
                next_state = rm_track::TRACK_TELE; 
                ROS_INFO("[FSM] Wide -> Tele");
            }
            else if (!device_state.wide_found) {
          
                next_state = rm_track::LOST_RAMP;
                ROS_WARN("[FSM] Wide Lost -> Ramp");
            }
            break;

        // --------------------------------------------------------
        // 状态 0: 长焦跟踪 (TRACK_TELE) 
        // --------------------------------------------------------
        case rm_track::TRACK_TELE:
            if (!device_state.tele_found) {
       
                if (device_state.wide_found) {
                    next_state = rm_track::TRACK_WIDE;
                    ROS_WARN("[FSM] Tele Lost -> Wide");
                }
                else {
                
                    next_state = rm_track::LOST_RAMP;
                    ROS_WARN("[FSM] Tele Lost -> Ramp");
                }
            }
            break;

        // --------------------------------------------------------
        // 状态 3: 丢失缓冲 (LOST_RAMP)
        // --------------------------------------------------------
        case rm_track::LOST_RAMP:
        
            if (device_state.tele_found) {
                next_state = rm_track::TRACK_TELE;
                ROS_INFO("[FSM] Ramp -> Tele");
            }
            else if (device_state.wide_found) {
                next_state = rm_track::TRACK_WIDE;
                ROS_INFO("[FSM] Ramp -> Wide");
            }
           
            else if (time_in_state > time_limits_) {
               
                if (device_state.radar_found) {
                    next_state = rm_track::TRACK_RADAR;
                    ROS_WARN("[FSM] Ramp Over -> Radar");
                }
                else {
                    next_state = rm_track::BLIND_SCAN;
                    ROS_WARN("[FSM] Ramp Over -> Blind");
                }
            }
            break;

        default:
            next_state = rm_track::BLIND_SCAN;
            break;
        }
        if (next_state != current_decision_state_)
        {
            current_decision_state_ = next_state;
            last_update_time_ = now; 
        }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    }

} // namespace rm_radarplugin