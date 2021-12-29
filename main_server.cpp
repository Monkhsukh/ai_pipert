#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <thread>
#include <future>

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
#include <iomanip>
#include "json.hpp"

using json = nlohmann::json;

#define PORT     8080
#define MAXLINE 1024


static int CHANNEL_CAPACITY = 10;

// GLOBAL VARIABLES
Scheduler sch(0);
PolledChannel<Mat> *pc_kf;
ScheduledChannel<frame_with_boxes *> *sc_kf;
KFTracker *kft;


void Initialize(){
    pc_kf = new PolledChannel<Mat>(sch.CreatePolledChannel<Mat>("OutTrack", CHANNEL_CAPACITY));
    kft = new KFTracker(pc_kf);
    sc_kf = new ScheduledChannel<frame_with_boxes *>(sch.CreateScheduledChannel<frame_with_boxes *>("Track", CHANNEL_CAPACITY, nullptr, bind(&KFTracker::Track, kft, placeholders::_1)));

}

vector<Rect> recieveClient(int sockfd, sockaddr_in *cliaddr)
{
    char buffer[MAXLINE];
    unsigned int len, n;
    len = sizeof(*cliaddr); 

    n = recvfrom(sockfd, (char *)buffer, MAXLINE,
                MSG_WAITALL, ( struct sockaddr *) cliaddr,
                &len);
    buffer[n] = '\0';
    
    // printf("Client : %s\n", buffer);

    json j_complete = json::parse(string(buffer));
    std::cout << std::setw(4) << j_complete << endl;



    vector<Rect> v_boxes = vector<Rect>();

    auto boxes = j_complete["boxes"];
    for(auto it : boxes){
        v_boxes.push_back(Rect(it["x"], it["y"], it["width"], it["height"]));
    }

    // cout<<"device:"<<j_complete["device"]<<" frame_id:"<<j_complete["frame_id"]
    // <<" detected_cnt:"<< v_boxes.size()<<endl;
    // cout<<j_complete["device"]<<"-"<<j_complete["frame_id"]<<"-"<< v_boxes.size()<<endl;

    return v_boxes;

}

void Run(char *argv)
{

    int sockfd;
    struct sockaddr_in servaddr, cliaddr;
    
    // Creating socket file descriptor
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    
    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));
    
    // Filling server information
    servaddr.sin_family = AF_INET; // IPv4
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(PORT);
    
    // Bind the socket with the server address
    if ( bind(sockfd, (const struct sockaddr *)&servaddr,
            sizeof(servaddr)) < 0 )
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }


    future<vector<Rect>> rf = std::async(std::launch::async, recieveClient, sockfd, &cliaddr); 
    future_status status = rf.wait_for(std::chrono::nanoseconds(1));
    
    Initialize();
    sch.Start();

    Mat frame, result_kf;
    bool isPushedKF = false;

    cv::VideoCapture cap;
    cap.open(argv);
    int fps_of_video = (int)cap.get(CAP_PROP_FPS);
    int time_to_wait = 1000 / fps_of_video;

    int frameCounter = 0;
    int tick = 0;
    int fps = fps_of_video;
    std::time_t timeBegin = std::time(0);

    vector<Rect> boxes = vector<Rect>();

    // while (true)
    for(int i=0; ;i++)
    {
        double time_start = (double)getTickCount();

        if(i%10 == 0)
        cout<<"->"<<i<<endl;

        cap.read(frame);
        if (frame.empty()) break;

        frameCounter++;

        std::time_t timeNow = std::time(0) - timeBegin;

        if (timeNow - tick >= 1)
        {
            tick++;
            fps = frameCounter;
            frameCounter = 0;
        }

        if(status == future_status::ready){
            boxes = rf.get();

            rf = std::async(std::launch::async, recieveClient, sockfd, &cliaddr); //thread_function
        }
        status = rf.wait_for(std::chrono::nanoseconds (1));
        
        /// KALMAN FILTER
        if (!isPushedKF)
        {
            pipert::Timer::Time time = pipert::Timer::time();
            frame_with_boxes *data = new frame_with_boxes();
            data->boxes = boxes;
            data->frame = frame.clone();
            PacketToFill<frame_with_boxes *> packet_to_fill1 = sc_kf->Acquire(time, data);
            packet_to_fill1.Push();
            isPushedKF = true;
        }
        
        PacketToProcess<Mat> packet_to_process_kf = pc_kf->Poll();
        if (!packet_to_process_kf.IsEmpty())
        {
            isPushedKF = false;
            result_kf = packet_to_process_kf.data();
        }


        if(!result_kf.empty()){
            
            cv::putText(frame, cv::format("Original FPS of Video=%d", fps_of_video), cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
            cv::putText(frame, cv::format("Average FPS=%d", fps), cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

            Mat combine;
            KFTracker::draw_boxes(boxes, frame);
            cv::Mat arr[] = {frame, result_kf};
            cv::hconcat(arr, 2, combine);

            cv::imshow("Display", combine);
        }

        // cv::imshow("Display", result_kf.empty()? frame : result_kf);
        
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;

        // wait for some time to correct FPS
        while (time_to_wait > ((double)getTickCount() - time_start) / getTickFrequency() * 1000)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    Run(argv[1]);
}
