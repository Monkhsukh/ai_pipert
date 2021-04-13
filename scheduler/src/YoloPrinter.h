#include "pipert/Scheduler.h"
#include "../../yolo/src/yolo.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class YoloPrinter
{
private:
    pipert::ScheduledChannel<Mat> *ch_to_write_;

public:
    YoloPrinter(pipert::ScheduledChannel<Mat> *ch_to_write);
    void Print(pipert::PacketToProcess<Mat> p);
};