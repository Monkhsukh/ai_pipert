#include "pipert/Scheduler.h"
#include "../../yolo/src/KalmanTracker.h"
#include "../../yolo/src/yolo.h"
#include <opencv2/opencv.hpp>
#include "../../utils/HungarianAlgorithm.h"

using namespace cv;

class KFTracker
{
private:
    pipert::PolledChannel<Mat> *pc_to_write_;
    vector<KalmanTracker> tracks;
    pipert::Timer::Time pre_time;
    vector<vector<double>> iou_mat;
    HungarianAlgorithm HungAlgo;
    vector<int> assignment;
    vector<pair<int, int>> matched;
    vector<int> unmatched_trks;
    vector<int> unmatched_dets;

public:
    const float IOU_TRSD = -0.2;

    KFTracker(pipert::PolledChannel<Mat> *pc_to_write);
    void Track(pipert::PacketToProcess<frame_with_boxes *> p);

    static void draw_boxes(vector<Rect> &boxes, Mat &frame);
    static float box_iou2(Rect &a, Rect &b);

    const bool VERBOSE = false;
};