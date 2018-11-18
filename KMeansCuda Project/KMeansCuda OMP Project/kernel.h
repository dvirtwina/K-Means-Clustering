#ifndef __KERNEL_H
#define __KERNEL_H

#include "Point.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

__global__ void setPointsForTimeIncrement(Point* dev_allPoints, float cosCalc, float sinCalc, int numOfPoints);
void setPointsDevice(Point* allPoints, float timeIncrement, float timeInterval);
void copyPointsToDevice(Point* allPoints, int numOfPoints);
void initDevice();
void releaseDeviceMemory();

#endif //__KERNEL_H