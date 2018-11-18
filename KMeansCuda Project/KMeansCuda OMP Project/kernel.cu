#include "kernel.h"
#include "Point.h"
#include <stdio.h>

const int MAX_THREADS_PER_BLOCK = 1024;
int numOfBlocks;
int dev_numOfPoints;
int chunkSize;
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

void setPointsDevice(Point* allPoints, float timeIncrement, float timeInterval)
{
	cudaError_t cudaStatus;
	//set the cos/sin calc for the threads ONE TIME ONLY.
	float cosCalc = (float)cos((2*acos(-1.0) *timeIncrement)/timeInterval);
	float sinCalc = (float)sin((2*acos(-1.0) *timeIncrement)/timeInterval);

	setPointsForTimeIncrement<<<numOfBlocks, MAX_THREADS_PER_BLOCK>>>(dev_allPoints, cosCalc, sinCalc, dev_numOfPoints);
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "setPointsForTimeIncrement launch failed: %s\n", cudaGetErrorString(cudaStatus));
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

	cudaStatus = cudaMemcpy(allPoints, dev_allPoints, dev_numOfPoints*sizeof(Point), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy from device to host failed!");
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
}

void copyPointsToDevice(Point* allPoints, int numOfPoints, int chunk)
{
	cudaError_t cudaStatus;
	dev_numOfPoints = numOfPoints;
	chunkSize = chunk;

	cudaStatus = cudaMalloc((void**)&dev_allPoints, dev_numOfPoints*sizeof(Point));
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc for dev_allPoints failed");
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
		fprintf(stderr, "cudaMemcpy from host to device failed!");
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
}

void initDevice()
{
	cudaError_t cudaStatus;
//	cudaDeviceProp deviceProp;

	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!");
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