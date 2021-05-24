#include "pipert/Scheduler.h"
#include "../../yolo/src/MyKalmanFilter.h"
#include "../../yolo/src/yolo.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class KFTracker
{
private:
    pipert::PolledChannel<Mat> *pc_to_write_;
    MyKalmanFilter kf;
    bool found;
    int notFoundCount;
    pipert::Timer::Time pre_time;

public:
    KFTracker(pipert::PolledChannel<Mat> *pc_to_write, MyKalmanFilter kf);
    void Track(pipert::PacketToProcess<frame_with_boxes *> p);

    MyKalmanFilter getMyKalmanFilter();
    void setMyKalmanFilter(MyKalmanFilter);
};