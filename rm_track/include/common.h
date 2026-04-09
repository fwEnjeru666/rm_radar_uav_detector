#pragma once

namespace rm_track
{
    // ModelType enum values must match the dynamic_reconfigure cfg file
    // cfg/ukf.cfg defines: CTRV=0, CV=1, CA=2
    enum ModelType
    {
        CTRV = 0,  // Constant Turn Rate and Velocity
        CV = 1,    // Constant Velocity
        CA = 2     // Constant Acceleration
    };

    enum TrackState
    {
        DETECTING = 0, // 1.first frame init
        // 2. predict and update
        TRACKING = 1, //normal predict and update
        TEMP_LOST = 2, // only predict without update
        LOST = 3 //reset
    };

}