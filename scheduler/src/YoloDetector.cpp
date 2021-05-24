#include "YoloDetector.h"

using namespace cv;

YoloDetector::YoloDetector(pipert::PolledChannel<frame_with_boxes *> *pc_to_write, Yolo _yolo)
    : pc_to_write_(pc_to_write), yolo(_yolo)
{
}

void YoloDetector::Detect(pipert::PacketToProcess<Mat> p)
{
    frame_with_boxes *frame = yolo.detect(p.data());
    pc_to_write_->Acquire(p.timestamp() + 10, frame);
}