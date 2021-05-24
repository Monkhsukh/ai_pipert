#include <opencv2/opencv.hpp>

using namespace cv;

#pragma once

class KalmanTracker
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
    Rect box;
    bool found = false;
    int hits = 0;
    int no_losses = 0;
    KalmanTracker(Rect box);
    ~KalmanTracker();
    void Draw(double dt, cv::Mat &frame);
    void UpdateMeas();
    bool GetFound();
};
