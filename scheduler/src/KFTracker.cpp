#include "KFTracker.h"

using namespace cv;

KFTracker::KFTracker(pipert::PolledChannel<Mat> *pc_to_write, MyKalmanFilter _kf)
    : pc_to_write_(pc_to_write), kf(_kf)
{
    pre_time = 0;
    found = false;
    notFoundCount = 0;
}

void KFTracker::Track(pipert::PacketToProcess<frame_with_boxes *> p)
{
    double dt = ((double)p.timestamp() - (double)pre_time) / (double)1000000;
    // cout << "Timestapm:" << dt << endl;
    frame_with_boxes *data = p.data();

    // KF TRACK
    if (found)
        kf.Draw(dt, data->frame);

    if (data->boxes.size() == 0)
    {
        notFoundCount++;
        // cout << "notFoundCount:" << notFoundCount << endl;
        if (notFoundCount >= 100)
            found = false;
    }
    else
    {
        notFoundCount = 0;
        found = kf.UpdateMeas(data->boxes, found);
    }
    this->pre_time = p.timestamp();

    pc_to_write_->Acquire(p.timestamp() + 10, data->frame);
}