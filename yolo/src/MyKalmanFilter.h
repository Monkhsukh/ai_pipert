#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#pragma once

class MyKalmanFilter
{
private:
    // >>>> Kalman Filter
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;
    unsigned int type = CV_32F;
    cv::KalmanFilter kf;
    cv::Mat state;
    cv::Mat meas;

public:
    MyKalmanFilter();
    ~MyKalmanFilter();
    void Draw(double dt, cv::Mat &frame);
    bool UpdateMeas(vector<Rect> &ret_boxes1, bool found);
};