#include "YoloPrinter.h"
#include <iostream>

using namespace cv;
using namespace std;

YoloPrinter::YoloPrinter(pipert::ScheduledChannel<Mat> *ch_to_write)
    : ch_to_write_(ch_to_write)
{
}

void YoloPrinter::Print(pipert::PacketToProcess<Mat> p)
{
    ch_to_write_->Acquire(p.timestamp() + 5, p.data());
}