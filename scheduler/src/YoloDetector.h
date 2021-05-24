#include "pipert/Scheduler.h"
#include "../../yolo/src/yolo.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class YoloDetector
{
private:
    pipert::PolledChannel<frame_with_boxes *> *pc_to_write_;
    Yolo yolo;

public:
    YoloDetector(pipert::PolledChannel<frame_with_boxes *> *pc_to_write, Yolo yolo);
    void Detect(pipert::PacketToProcess<Mat> p);
    Yolo getYolo();
    void setYolo(Yolo);
};