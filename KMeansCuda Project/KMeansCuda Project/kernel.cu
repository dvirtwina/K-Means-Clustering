#include "kernel.h"
#include "Point.h"
#include <stdio.h>
#include <time.h>
using namespace std;

const int MAX_THREADS_PER_BLOCK = 1024;
int numOfBlocks;
int dev_numOfPoints;
Point* dev_allPoints;
float* dev_distancesArr;

/** each thread in the GPU executes a distance measurment from the given point to the thread's assigned point.*/
__global__ void setDistancesForPoint(Point* dev_allPoints, int dev_numOfPoints, float* dev_distancesArr, float pntX, float pntY)
{
	float x,y, dist;
	int currentThread = threadIdx.x;
	int currentBlock = blockIdx.x;
	//set position of the thread.
	int pos = currentBlock*MAX_THREADS_PER_BLOCK + currentThread;

	if(pos < dev_numOfPoints)
	{
		x = (pntX - dev_allPoints[pos].x)*(pntX - dev_allPoints[pos].x);
		y = (pntY - dev_allPoints[pos].y)*(pntY - dev_allPoints[pos].y);
		dist = sqrt(x + y);
		dev_distancesArr[pos] = dist;
	}
}

/**calculate the distance from a given point to every other point in the array.*/
void cudaDistancesOfPoint(Point* allPoints, float* distances, float pntX, float pntY)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&dev_distancesArr, dev_numOfPoints*sizeof(float));
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc for dev_distancesArr failed\n");
		releaseDeviceMemory();
		return;
	}
	setDistancesForPoint<<<numOfBlocks, MAX_THREADS_PER_BLOCK>>>(dev_allPoints, dev_numOfPoints, dev_distancesArr, pntX, pntY);

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "setDistancesForPoint launch failed: %s\n", cudaGetErrorString(cudaStatus));
		releaseDeviceMemory();
		return;
	}

	cudaStatus = cudaDeviceSynchronize(); 
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDevice Synchronize returned error code %d\n", cudaStatus);
		releaseDeviceMemory();
		return;
	}

	cudaStatus = cudaMemcpy(distances, dev_distancesArr, dev_numOfPoints*sizeof(float), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy from device to host failed!\n");
		releaseDeviceMemory();
		return;
	}

	cudaFree(dev_distancesArr);
}

/** Copy an array of points to the GPU device.*/
void copyPointsToDevice(Point* allPoints, int numOfPoints)
{
	cudaError_t cudaStatus;
	dev_numOfPoints = numOfPoints;

	cudaStatus = cudaMalloc((void**)&dev_allPoints, dev_numOfPoints*sizeof(Point));
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc for dev_allPoints failed\n");
		releaseDeviceMemory();
		return;
	}

	cudaStatus = cudaMemcpy(dev_allPoints, allPoints, dev_numOfPoints*sizeof(Point), cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy from host to device failed!\n");
		releaseDeviceMemory();
		return;
	}

	numOfBlocks = dev_numOfPoints/MAX_THREADS_PER_BLOCK;
	if(dev_numOfPoints%MAX_THREADS_PER_BLOCK != 0)
		numOfBlocks += 1;
}

/** Instantiating GPU device.*/
void initDevice()
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!\n");
		releaseDeviceMemory();
		return;
	}
}

void releaseDeviceMemory()
{
	cudaFree(dev_allPoints);
}