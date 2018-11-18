#include "kernel.h"
#include "Point.h"
#include <stdio.h>
using namespace std;

const int MAX_THREADS_PER_BLOCK = 1024;
int numOfBlocks;
int dev_numOfPoints;
Point* dev_allPoints;
float* dev_distancesArr;

__global__ void setPointsForTimeIncrement(Point* dev_allPoints, float cosCalc, float sinCalc, int numOfPoints)
{
	//int currentThread = threadIdx.x;
	//int currentBlock = blockIdx.x;
	////setting the position of the thread.
	//int pos = currentBlock*MAX_THREADS_PER_BLOCK + currentThread;
	//if(pos < numOfPoints)
	//{
	//	dev_allPoints[pos].x = (float)dev_allPoints[pos].a + (dev_allPoints[pos].radius*cosCalc);
	//	dev_allPoints[pos].y = (float)dev_allPoints[pos].b + (dev_allPoints[pos].radius*sinCalc);
	//}
}

__global__ void setDistancesForPoint(Point* dev_allPoints, int dev_numOfPoints, float* dev_distancesArr, float pntX, float pntY)
{
	float x,y, dist;
	int currentThread = threadIdx.x;
	int currentBlock = blockIdx.x;
	//setting the position of the thread.
	int pos = currentBlock*MAX_THREADS_PER_BLOCK + currentThread;
	if(pos < dev_numOfPoints)
	{
		x = (pntX - dev_allPoints[pos].x)*(pntX - dev_allPoints[pos].x);
		y = (pntY - dev_allPoints[pos].y)*(pntY - dev_allPoints[pos].y);
		dist = sqrt(x + y);
		dev_distancesArr[pos] = dist;
	}
}

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
//	cout << "CUDA: pntX = " << pntX << "pntY = " << pntY << endl;
	setDistancesForPoint<<<numOfBlocks, MAX_THREADS_PER_BLOCK>>>(dev_allPoints, dev_numOfPoints, dev_distancesArr, pntX, pntY);
	
	//set the cos/sin calc for the threads ONE TIME ONLY.
	/*float cosCalc = (float)cos((2*acos(-1.0) *timeIncrement)/timeInterval);
	float sinCalc = (float)sin((2*acos(-1.0) *timeIncrement)/timeInterval);*/

//	setPointsForTimeIncrement<<<numOfBlocks, MAX_THREADS_PER_BLOCK>>>(dev_allPoints, cosCalc, sinCalc, dev_numOfPoints);
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

	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDevice Synchronize returned error code %d\n", cudaStatus);
		releaseDeviceMemory();
		return;
	}

	cudaFree(dev_distancesArr);
}

void copyPointsToDevice(Point* allPoints, int numOfPoints)
{
//	printf("CUDA: copyPointsToDevice\n");fflush(stdout);
	cudaError_t cudaStatus;
	dev_numOfPoints = numOfPoints;

	cudaStatus = cudaMalloc((void**)&dev_allPoints, dev_numOfPoints*sizeof(Point));
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc for dev_allPoints failed\n");
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

	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDevice Synchronize returned error code %d\n", cudaStatus);
		releaseDeviceMemory();
		return;
	}

	///////////// TO BE REMOVED
	//float x = 1.0;
	//float y = 2.0;
	//printf("dev_allPoints[0]:\tx=%f\ty=%f\n", /*dev_allPoints[0].x*/x, /*dev_allPoints[0].y*/y);fflush(stdout);
	///////////
}

void initDevice()
{
	cudaError_t cudaStatus;
//	cudaDeviceProp deviceProp;

	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!\n");
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

//	cudaGetDeviceProperties(&deviceProp, 0);
//	maxNumberOfThreadPerBlock = deviceProp.maxThreadsPerBlock;
}

void releaseDeviceMemory()
{
	cudaFree(dev_allPoints);
}