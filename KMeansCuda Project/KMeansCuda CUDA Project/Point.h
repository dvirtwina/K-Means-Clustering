#ifndef __POINT_H
#define __POINT_H

#include <math.h>
#include <vector>
using namespace std;

class Point
{
private:
	int index;
	int nearestClusterIndex;
	int numberOfClusters;
	float* distancesFromClusters;

public:
	float x;
	float y;

	void setPoint(int index, float x, float y, int numberOfClusters)
	{
		this->index = index;
		nearestClusterIndex = 0;
		this->x = x;
		this->y = y;
		this->numberOfClusters = numberOfClusters;
		distancesFromClusters = new float[numberOfClusters]();
//		cout << "Point created: index=" << index << "\tx=" << x << "\ty=" << y << endl;
	}
	void setPoint(int index, float x, float y, int numberOfClusters, int nearestClusterIndex)
	{
		this->index = index;
		this->nearestClusterIndex = nearestClusterIndex;
		this->x = x;
		this->y = y;
		this->numberOfClusters = numberOfClusters;
		distancesFromClusters = new float[numberOfClusters]();
//		cout << "Point created: index=" << index << "\tx=" << x << "\ty=" << y << "\tNearestCluster = " << nearestClusterIndex << endl;
	}
	~Point() { delete[] distancesFromClusters; }

	void setNearestCluster()
	{
		float min = distancesFromClusters[0];
		int index = 0;
		for (int i = 0; i < numberOfClusters; i++)
		{
			if (distancesFromClusters[i] < min)
			{
				min = distancesFromClusters[i];
				index = i;
			}
		}
		nearestClusterIndex = index;
	}

	int getNearestCluster()
	{
		return nearestClusterIndex;
	}

	void setDistanceFromClusters(float dist, int index)
	{
		distancesFromClusters[index] = dist;
	}

	float getDistanceFromClusterIndex(int index)
	{
		return distancesFromClusters[index];
	}

	int getIndex() { return index; }
	float getX() { return x; }
	float getY() { return y; }
	void setNumOfClusters(int numOfClusters)
	{
		this->numberOfClusters = numOfClusters;
	}
//	void print() const { cout << "POINT: (" << index << ") " << x << "\t" << y << "\tNearestCluster = " << nearestClusterIndex << endl; };
};



#endif //__POINT_H