#ifndef __KERNEL_H
#define __KERNEL_H

#include "Point.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

__global__ void setDistancesForPoint(float pntX, float pntY);
void cudaDistancesOfPoint(Point* allPoints, float* distances, float pntX, float pntY);
void copyPointsToDevice(Point* allPoints, int numOfPoints);
void initDevice();
void releaseDeviceMemory();

#endif //__KERNEL_H