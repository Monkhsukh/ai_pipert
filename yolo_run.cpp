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

using namespace std;
using namespace cv;
using namespace dnn;

static String CONFIG = "./yolo/cfg/yolov3.cfg";
static String MODEL = "./yolo/cfg/yolov3.weights";
static String CLASSESFILE = "./yolo/cfg/coco_copy.names";

static int CHANNEL_CAPACITY = 10;

// Global variables
Yolo yolo(CONFIG, MODEL, CLASSESFILE);


void RunYolo(char *argv)
{
	Mat frame;
	frame_with_boxes *result = nullptr;

	cv::VideoCapture cap;
	cap.open(argv);
	while(true)
	{
		cap.read(frame);
		if(frame.empty())
			break;
		
		
		result = yolo.detect(frame);


		cv:namedWindow("Display", cv::WINDOW_NORMAL);
        cv::resizeWindow("Display", 1024, 1024); 
        cv::imshow("Display", result->frame);

        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
	}
}

int main(int argc, char **argv)
{
	RunYolo(argv[1]);
}
