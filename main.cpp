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

using namespace std;
using namespace cv;
using namespace dnn;

#include "yolo/src/yolo.h"
#include "yolo/src/MyKalmanFilter.h"

#include "scheduler/src/YoloDetector.h"
#include "scheduler/src/YoloPrinter.h"
#include "scheduler/src/KFTracker.h"

#include "pipert/Scheduler.h"
#include "pipert/Profiler.h"

#include "utils/HungarianAlgorithm.h"

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
MyKalmanFilter kf = MyKalmanFilter();

HungarianAlgorithm HungAlgo;

void Initialize()
{
    pc_tiny = new PolledChannel<frame_with_boxes *>(sch.CreatePolledChannel<frame_with_boxes *>("OutChannel", CHANNEL_CAPACITY));
    pc = new PolledChannel<frame_with_boxes *>(sch.CreatePolledChannel<frame_with_boxes *>("OutChannelYolo", CHANNEL_CAPACITY));
    yd_tiny = new YoloDetector(pc_tiny, yolo_tiny);
    yd = new YoloDetector(pc, yolo);
    sc_tiny = new ScheduledChannel<Mat>(sch.CreateScheduledChannel<Mat>("DetectorChannel", CHANNEL_CAPACITY, nullptr, bind(&YoloDetector::Detect, yd_tiny, placeholders::_1)));
    sc = new ScheduledChannel<Mat>(sch.CreateScheduledChannel<Mat>("DetectorChannelYolo", CHANNEL_CAPACITY, nullptr, bind(&YoloDetector::Detect, yd, placeholders::_1)));

    pc_kf = new PolledChannel<Mat>(sch.CreatePolledChannel<Mat>("Out", CHANNEL_CAPACITY));
    kft = new KFTracker(pc_kf, kf);
    sc_kf = new ScheduledChannel<frame_with_boxes *>(sch.CreateScheduledChannel<frame_with_boxes *>("Track", CHANNEL_CAPACITY, nullptr, bind(&KFTracker::Track, kft, placeholders::_1)));
}

void draw_boxes(vector<Rect> &boxes, Mat &frame);

float box_iou2(Rect &a, Rect &b)
{
    int w_intsec = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x));
    int h_intsec = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y));
    int s_intsec = w_intsec * h_intsec;
    int s_a = a.height * a.width;
    int s_b = b.height * b.width;
    float ret = float(s_intsec) / float(s_a + s_b - s_intsec);
    // cout << "s_intsec:" << s_intsec << endl;
    // cout << "iou:" << ret << endl;
    return ret * -1;
}

class MyTracker
{
public:
    int id = 0;
    int hits = 0;
    int no_losses = 0;
};

vector<vector<double>> iou_mat;
vector<pair<int, int>> matched;
vector<int> assignment;
vector<MyTracker> t_list;
vector<Rect> track_list;
vector<int> unmatched_trks;
vector<int> unmatched_dets;

int main(int argc, char **argv)
{
    Mat frame, row1, row2, combine, result_kf, frame_clone;
    frame_with_boxes *result_tiny, *result_yolo;
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
            // Save old detections to track list
            if (result_tiny != nullptr)
            {
                track_list = result_tiny->boxes;
                for (auto i = 0; i < result_tiny->boxes.size(); i++)
                    t_list.push_back(MyTracker());
            }

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

        float iou_trsd = 0.2 * (-1);
        cout << track_list.size() << endl;
        if (result_tiny != nullptr && track_list.size() > 0)
        {
            int t_num = track_list.size();
            int d_num = result_tiny->boxes.size();

            if (d_num > 0)
            {
                cout << "track:" << t_num << "\t"
                     << "det:" << d_num << endl;
                // Create IOU matrix
                iou_mat.clear();
                for (int i = 0; i < t_num; i++)
                {
                    vector<double> row;
                    for (int j = 0; j < d_num; j++)
                        row.push_back(box_iou2(track_list[i], result_tiny->boxes[j]));
                    iou_mat.push_back(row);
                }

                for (int i = 0; i < t_num; i++)
                {
                    for (int j = 0; j < d_num; j++)
                        cout << iou_mat[i][j] << "\t";
                    cout << endl;
                }

                HungAlgo.Solve(iou_mat, assignment);
                for (unsigned int x = 0; x < iou_mat.size(); x++)
                    cout << x << "," << assignment[x] << "\t";
                cout << endl;

                matched.clear();
                unmatched_dets.clear();
                unmatched_trks.clear();

                for (int i = 0; i < d_num; i++)
                    if (find(assignment.begin(), assignment.end(), i) == assignment.end())
                        unmatched_dets.push_back(i);

                for (auto i = 0; i < t_num; i++)
                    if (assignment[i] >= 0)
                    {
                        if (iou_mat[i][assignment[i]] < iou_trsd) // THRESHOLD is MINUS!
                            matched.push_back(make_pair(i, assignment[i]));
                        else
                        {
                            unmatched_dets.push_back(assignment[i]);
                            unmatched_trks.push_back(i);
                        }
                    }
                    else
                        unmatched_trks.push_back(i);

                cout << "un tracks" << endl;
                for (auto a : unmatched_trks)
                    cout << a << endl;
                cout << "un dets" << endl;
                for (auto b : unmatched_dets)
                    cout << b << endl;
                cout << "matched" << endl;
                for (auto c : matched)
                    cout << c.first << ":" << c.second << endl;

                // Matched
                if (matched.size() > 0)
                {
                    for (auto c : matched)
                    {
                        t_list[c.first].hits++;
                        t_list[c.first].no_losses = 0;
                    }
                }

                //UnMatched Tracks
                for (auto t : unmatched_trks)
                    t_list[t].no_losses++;

                // DELETE OLD TRACKS
                for (auto i = t_list.size(); i <= 0; i--)
                    if (t_list[i].no_losses > 2)
                    {
                        t_list.erase(t_list.begin() + i);
                        track_list.erase(track_list.begin() + i);
                    }
            }
        }

        /// SHOW VIDEO
        if (result_tiny != nullptr && result_yolo != nullptr)
        {
            cv::Mat arr[] = {frame, result_tiny->frame, result_yolo->frame};
            cv::hconcat(arr, 3, row1);

            draw_boxes(result_tiny->boxes, frame);
            draw_boxes(result_tiny->boxes, result_kf);
            draw_boxes(result_yolo->boxes, frame_clone);

            cv::Mat arr2[] = {frame, result_kf, frame_clone};
            cv::hconcat(arr2, 3, row2);
            cv::Mat arr3[] = {row1, row2};
            cv::vconcat(arr3, 2, combine);
            cv::imshow("Display", combine);
        }

        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }
    sch.Stop();
}

void draw_boxes(vector<Rect> &boxes, Mat &frame)
{
    for (size_t i = 0; i < boxes.size(); i++)
    {
        Rect box = boxes[i];
        rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);
    }
};
