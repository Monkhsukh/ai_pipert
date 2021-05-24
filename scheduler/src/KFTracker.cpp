#include "KFTracker.h"

using namespace cv;
using namespace std;

KFTracker::KFTracker(pipert::PolledChannel<Mat> *pc_to_write)
    : pc_to_write_(pc_to_write)
{
    pre_time = 0;
}

void KFTracker::Track(pipert::PacketToProcess<frame_with_boxes *> p)
{
    double dt = ((double)p.timestamp() - (double)pre_time) / (double)1000000;
    frame_with_boxes *data = p.data();

    // Hungarian algo
    int t_num = tracks.size();
    int d_num = data->boxes.size();

    matched.clear();
    unmatched_dets.clear();
    unmatched_trks.clear();
    iou_mat.clear();
    assignment.clear();

    if (d_num > 0 && t_num > 0)
    {
        for (int i = 0; i < t_num; i++)
        {
            vector<double> row;
            for (int j = 0; j < d_num; j++)
                row.push_back(box_iou2(tracks[i].box, data->boxes[j]));
            iou_mat.push_back(row);
        }
        HungAlgo.Solve(iou_mat, assignment);

        if (VERBOSE)
        {
            // PRINT IOU_MAT
            cout << "IOU" << endl;
            for (int i = 0; i < t_num; i++)
            {
                for (int j = 0; j < d_num; j++)
                    cout << iou_mat[i][j] << "\t";
                cout << endl;
            }

            for (unsigned int x = 0; x < iou_mat.size(); x++)
                cout << x << "," << assignment[x] << "\t";
            cout << endl;
        } // END VERBOSE
    }

    // UNMATCHED DETECTIONS
    for (int i = 0; i < d_num; i++)
        if (find(assignment.begin(), assignment.end(), i) == assignment.end())
            unmatched_dets.push_back(i);

    // UNMATCHED TRACKS
    for (auto i = 0; i < t_num; i++)
        if (i >= assignment.size())
            unmatched_trks.push_back(i);
        else
        {
            if (assignment[i] >= 0)
            {
                if (iou_mat[i][assignment[i]] < IOU_TRSD) // THRESHOLD is MINUS!
                    matched.push_back(make_pair(i, assignment[i]));
                else
                {
                    unmatched_dets.push_back(assignment[i]);
                    unmatched_trks.push_back(i);
                }
            }
            else
                unmatched_trks.push_back(i);
        }

    if (VERBOSE)
    {
        cout << "unmatched tracks" << endl;
        for (auto a : unmatched_trks)
            cout << a << endl;
        cout << "unmatched dets" << endl;
        for (auto b : unmatched_dets)
            cout << b << endl;
        cout << "matched" << endl;
        for (auto c : matched)
            cout << c.first << ":" << c.second << endl;
    } // END VERBOSE

    // Matched
    if (matched.size() > 0)
        for (auto c : matched)
        {
            tracks[c.first].hits++;
            tracks[c.first].no_losses = 0;
            tracks[c.first].box = data->boxes[c.second];
        }

    //UnMatched Tracks
    for (auto t : unmatched_trks)
        tracks[t].no_losses++;

    //UnMached Detects
    KalmanTracker *kt;
    for (auto d : unmatched_dets)
    {
        kt = new KalmanTracker(data->boxes[d]);
        tracks.push_back(*kt);
    }

    // DELETE OLD TRACKS
    for (int i = tracks.size() - 1; i >= 0; i--)
    {
        if (tracks[i].no_losses > 4)
            tracks.erase(tracks.begin() + i);
    }

    // KF TRACK
    for (auto &t : tracks)
    {
        if (VERBOSE)
        {
            cout << "loss:" << t.no_losses << "\t";
            cout << "hits:" << t.hits << endl;
            cout << "found:" << t.GetFound() << endl;
        } // END VERBOSE

        if (t.found || t.hits > 2)
            t.Draw(dt, data->frame);

        t.UpdateMeas();
    }

    this->pre_time = p.timestamp();
    pc_to_write_->Acquire(p.timestamp() + 10, data->frame);
}

void KFTracker::draw_boxes(vector<Rect> &boxes, Mat &frame)
{
    for (size_t i = 0; i < boxes.size(); i++)
    {
        Rect box = boxes[i];
        rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);
    }
};

float KFTracker::box_iou2(Rect &a, Rect &b)
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