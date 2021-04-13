#include <fstream>
#include <sstream>
#include <iostream>

// Required for dnn modules.
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

#pragma once

class Yolo
{
private:
    vector<string> classes;
    String configuration;
    String model;
    String classesFile;
    Net net;

public:
    // confidence threshold
    static constexpr float conf_threshold = 0.5;
    // nms threshold
    static constexpr float nms = 0.4;
    static const int width = 416;
    static const int height = 416;

    Yolo(String configuration, String model, String classesFile);

    // remove unnecessary bounding boxes
    void remove_box(Mat &frame, const vector<Mat> &out);

    // draw bounding boxes
    void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat &frame);

    // get output layers
    vector<String> getOutputsNames(const Net &net);

    Mat detect(Mat frame);
};