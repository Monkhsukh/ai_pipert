#include <opencv2/opencv.hpp>
#include <iostream>
#include "KalmanTracker.h"

using namespace std;

KalmanTracker::KalmanTracker(Rect _box)
{
    box = _box;
    kf = cv::KalmanFilter(stateSize, measSize, contrSize, type);
    state = cv::Mat(stateSize, 1, type); // [x,y,v_x,v_y,w,h]
    meas = cv::Mat(measSize, 1, type);   // [z_x,z_y,z_w,z_h]

    cv::setIdentity(kf.transitionMatrix);
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
}

KalmanTracker::~KalmanTracker()
{
}

bool KalmanTracker::GetFound()
{
    return found;
}

void KalmanTracker::Draw(double dT, cv::Mat &frame)
{
    // >>>> Matrix A
    kf.transitionMatrix.at<float>(2) = dT;
    kf.transitionMatrix.at<float>(9) = dT;

    // cout << "dT:" << endl
    //      << dT << endl;

    state = kf.predict();
    // cout << "State post:" << endl
    //      << state << endl;

    cv::Rect predRect;
    predRect.width = state.at<float>(4);
    predRect.height = state.at<float>(5);
    predRect.x = state.at<float>(0) - predRect.width / 2;
    predRect.y = state.at<float>(1) - predRect.height / 2;

    cv::Point center;
    center.x = state.at<float>(0);
    center.y = state.at<float>(1);
    cv::circle(frame, center, 2, CV_RGB(255, 0, 0), -1);

    cv::rectangle(frame, predRect, CV_RGB(255, 0, 0), 2);
}

void KalmanTracker::UpdateMeas()
{
    meas.at<float>(0) = box.x + box.width / 2;
    meas.at<float>(1) = box.y + box.height / 2;
    meas.at<float>(2) = (float)box.width;
    meas.at<float>(3) = (float)box.height;

    if (this->found) // First detection!
    {
        kf.correct(meas); // Kalman Correction
    }
    else
    {
        // >>>> Initialization
        kf.errorCovPre.at<float>(0) = 1; // px
        kf.errorCovPre.at<float>(7) = 1; // px
        kf.errorCovPre.at<float>(14) = 1;
        kf.errorCovPre.at<float>(21) = 1;
        kf.errorCovPre.at<float>(28) = 1; // px
        kf.errorCovPre.at<float>(35) = 1; // px

        state.at<float>(0) = meas.at<float>(0);
        state.at<float>(1) = meas.at<float>(1);
        state.at<float>(2) = 0;
        state.at<float>(3) = 0;
        state.at<float>(4) = meas.at<float>(2);
        state.at<float>(5) = meas.at<float>(3);
        // <<<< Initialization

        kf.statePost = state;

        this->found = true;
    }

    // cout << "Measure matrix:" << endl
    //      << meas << endl;
}