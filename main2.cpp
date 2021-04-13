#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>

// Required for dnn modules.
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

#include "yolo/src/yolo.h"
#include "scheduler/src/YoloDetector.h"
#include "scheduler/src/YoloPrinter.h"
#include "pipert/Scheduler.h"
#include "pipert/Profiler.h"

vector<string> classes;

int main(int argc, char **argv)
{
    String configuration = "./yolo/cfg/yolov3-tiny.cfg";
    String model = "./yolo/cfg/yolov3-tiny.weights";
    string classesFile = "./yolo/cfg/coco.names";
    Yolo yolo(configuration, model, classesFile);

    // pipert::Scheduler sch(0, pipert::Profiler("file:profilerlog.txt"));
    pipert::Scheduler sch(0, pipert::Profiler("udp:127.0.0.1:8000"));

    int channel_capacity = 10;
    pipert::PolledChannel<Mat> pc =
        sch.CreatePolledChannel<Mat>("OutChannel", channel_capacity);

    YoloDetector yd(&pc, yolo);
    pipert::ScheduledChannel<Mat> sc2 =
        sch.CreateScheduledChannel<Mat>("DetectorChannel", channel_capacity, nullptr, bind(&YoloDetector::Detect, &yd, placeholders::_1));

    YoloPrinter yp(&sc2);
    pipert::ScheduledChannel<Mat> sc1 =
        sch.CreateScheduledChannel<Mat>("PrinterChannel", channel_capacity, nullptr, bind(&YoloPrinter::Print, &yp, placeholders::_1));

    sch.Start();
    pipert::Timer::Time time = pipert::Timer::time();

    Mat frame;
    cv::VideoCapture cap;
    cap.open(argv[1]);

    while (true)
    {
        cap.read(frame);
        if (frame.empty())
            break;

        pipert::PacketToFill<Mat> packet_to_fill =
            sc1.Acquire(time, frame.clone());
        packet_to_fill.Push();

        pipert::PacketToProcess<Mat> packet_to_process = pc.Poll();
        while (packet_to_process.IsEmpty())
        {
            packet_to_process = pc.Poll();
        }

        sch.GetProfiler().GatherNSend();

        cv::imshow("Display", packet_to_process.data());

        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }
    sch.Stop();
}