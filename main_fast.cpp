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

#include "pipert/Scheduler.h"
#include "pipert/Profiler.h"

using namespace std;

class FrameTranslator
{
private:
    pipert::PolledChannel<Mat> *ch_to_write_;

public:
    FrameTranslator(pipert::PolledChannel<Mat> *ch_to_write) : ch_to_write_(ch_to_write) {}

    void Translator(pipert::PacketToProcess<Mat> p)
    {
        // pipert::Timer::Time time = pipert::Timer::time();
        // float sec = (time - p.timestamp());
        // cout << sec << endl;
        ch_to_write_->Acquire(p.timestamp() + 5, p.data());
    }
};

int main(int argc, char **argv)
{
    // pipert::Scheduler sch(0, pipert::Profiler("file:fast_profilerlog.txt"));
    pipert::Scheduler sch(0, pipert::Profiler("udp:127.0.0.1:8000"));

    int channel_capacity = 10;
    pipert::PolledChannel<Mat> pc =
        sch.CreatePolledChannel<Mat>("OutChannel", channel_capacity);

    FrameTranslator ft(&pc);
    pipert::ScheduledChannel<Mat> sc =
        sch.CreateScheduledChannel<Mat>("TranslatorChannel", channel_capacity, nullptr, bind(&FrameTranslator::Translator, &ft, placeholders::_1));

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
        // time = pipert::Timer::time();
        pipert::PacketToFill<Mat> packet_to_fill =
            sc.Acquire(time, frame.clone());
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