#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <vector>

// Required for dnn modules.
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "yolo/src/yolo.h"
#include "yolo/src/MyKalmanFilter.h"

#include "scheduler/src/YoloDetector.h"
#include "scheduler/src/YoloPrinter.h"
#include "scheduler/src/KFTracker.h"

#include "pipert/Scheduler.h"
#include "pipert/Profiler.h"

#include "utils/HungarianAlgorithm.h"

using namespace std;
using namespace cv;
using namespace dnn;
using namespace pipert;

static String CONFIG_TINY = "./yolo/cfg/yolov3-tiny.cfg";
static String MODEL_TINY = "./yolo/cfg/yolov3-tiny.weights";
static String CLASSESFILE_TINY = "./yolo/cfg/coco.names";

static String CONFIG = "./yolo/cfg/yolov3.cfg";
static String MODEL = "./yolo/cfg/yolov3.weights";
static String CLASSESFILE = "./yolo/cfg/coco_copy.names";

const static char *PROFILER_PATH = "udp:127.0.0.1:8000";

static int CHANNEL_CAPACITY = 10;

// GLOBAL VARIABLES
Scheduler sch(0, Profiler(PROFILER_PATH));
PolledChannel<frame_with_boxes *> *pc_tiny;
PolledChannel<frame_with_boxes *> *pc;
PolledChannel<Mat> *pc_kf;
ScheduledChannel<Mat> *sc_tiny;
ScheduledChannel<Mat> *sc;
ScheduledChannel<frame_with_boxes *> *sc_kf;
Yolo yolo_tiny(CONFIG_TINY, MODEL_TINY, CLASSESFILE_TINY);
Yolo yolo(CONFIG, MODEL, CLASSESFILE);
YoloDetector *yd_tiny;
YoloDetector *yd;

KFTracker *kft;

void Initialize()
{
    pc_tiny = new PolledChannel<frame_with_boxes *>(sch.CreatePolledChannel<frame_with_boxes *>("OutChannel", CHANNEL_CAPACITY));
    pc = new PolledChannel<frame_with_boxes *>(sch.CreatePolledChannel<frame_with_boxes *>("OutChannelYolo", CHANNEL_CAPACITY));
    yd_tiny = new YoloDetector(pc_tiny, yolo_tiny);
    yd = new YoloDetector(pc, yolo);
    sc_tiny = new ScheduledChannel<Mat>(sch.CreateScheduledChannel<Mat>("DetectorChannel", CHANNEL_CAPACITY, nullptr, bind(&YoloDetector::Detect, yd_tiny, placeholders::_1)));
    sc = new ScheduledChannel<Mat>(sch.CreateScheduledChannel<Mat>("DetectorChannelYolo", CHANNEL_CAPACITY, nullptr, bind(&YoloDetector::Detect, yd, placeholders::_1)));

    pc_kf = new PolledChannel<Mat>(sch.CreatePolledChannel<Mat>("Out", CHANNEL_CAPACITY));
    kft = new KFTracker(pc_kf);
    sc_kf = new ScheduledChannel<frame_with_boxes *>(sch.CreateScheduledChannel<frame_with_boxes *>("Track", CHANNEL_CAPACITY, nullptr, bind(&KFTracker::Track, kft, placeholders::_1)));
}

int main(int argc, char **argv)
{
    Mat frame, row1, row2, combine, result_kf, frame_clone;
    frame_with_boxes *result_tiny = nullptr, *result_yolo = nullptr;
    bool isPushed = false, isPushedYolo = false, isPushedKF = false;

    Initialize();
    sch.Start();

    cv::VideoCapture cap;
    cap.open(argv[1]);
    while (true)
    {
        pipert::Timer::Time time = pipert::Timer::time();

        cap.read(frame);
        if (frame.empty())
            break;
        frame.copyTo(frame_clone);

        /// KALMAN FILTER
        if (!isPushedKF && result_tiny != nullptr)
        {
            frame_with_boxes *data = new frame_with_boxes();
            data->boxes = result_tiny->boxes;
            data->frame = frame.clone();
            PacketToFill<frame_with_boxes *> packet_to_fill1 = sc_kf->Acquire(time, data);
            packet_to_fill1.Push();
            isPushedKF = true;
        }
        PacketToProcess<Mat> packet_to_process1 = pc_kf->Poll();
        if (!packet_to_process1.IsEmpty())
        {
            isPushedKF = false;
            result_kf = packet_to_process1.data();
        }

        /// YOLO TINY
        if (!isPushed)
        {
            pipert::PacketToFill<Mat> packet_to_fill =
                sc_tiny->Acquire(time, frame.clone());
            packet_to_fill.Push();
            isPushed = true;
        }
        pipert::PacketToProcess<frame_with_boxes *> packet_to_process = pc_tiny->Poll();
        if (!packet_to_process.IsEmpty())
        {
            isPushed = false;
            result_tiny = packet_to_process.data();
        }

        /// YOLO
        if (!isPushedYolo)
        {
            pipert::PacketToFill<Mat> packet_to_fill_yolo =
                sc->Acquire(time, frame.clone());
            packet_to_fill_yolo.Push();
            isPushedYolo = true;
        }
        pipert::PacketToProcess<frame_with_boxes *> packet_to_process_yolo = pc->Poll();
        if (!packet_to_process_yolo.IsEmpty())
        {
            isPushedYolo = false;
            result_yolo = packet_to_process_yolo.data();
        }

        /// SHOW VIDEO
        if (result_tiny != nullptr && result_yolo != nullptr)
        {
            cv::Mat arr[] = {frame, result_tiny->frame, result_yolo->frame};
            cv::hconcat(arr, 3, row1);

            KFTracker::draw_boxes(result_tiny->boxes, frame);
            KFTracker::draw_boxes(result_yolo->boxes, frame_clone);

            cv::Mat arr2[] = {result_kf, frame, frame_clone};
            cv::hconcat(arr2, 3, row2);
            cv::Mat arr3[] = {row1, row2};
            cv::vconcat(arr3, 2, combine);
            cv::imshow("Display", combine);
        }

        sch.GetProfiler().GatherNSend();

        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }
    sch.Stop();
}
