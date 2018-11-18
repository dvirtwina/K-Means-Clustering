#ifndef __CLUSTER_H
#define __CLUSTER_H

#include <math.h>
#include <vector>
using namespace std;

class Cluster
{
private:
	int index;
	bool changed;
	float x;
	float y;
	float diameter;
	int numberOfClusters;
	float* distancesToOtherClusters;
	
public:
	void setCluster(int index, float x, float y, int numberOfClusters)
	{
		changed = true; /////
		this->index = index;
		this->x = x;
		this->y = y;
		this->numberOfClusters = numberOfClusters;
		distancesToOtherClusters = new float[numberOfClusters];
	};
	~Cluster() { delete[] distancesToOtherClusters; }

	/**Calculates the distance from this cluster to a given point p.*/
	float getDistanceFromPoint(Point& p)
	{
		float pX = p.getX();
		float pY = p.getY();
		double defX = (x - pX)*(x - pX);
		double defY = (y - pY)*(y - pY);

		return (float)sqrt(defX + defY);
	};

	/** Calculates the distance from to cluster to all other clusters.*/
	void setDistancesToOtherClusters(Cluster* clusters)
	{
		for (int clust = 0; clust < numberOfClusters; clust++)
		{
			if (clust == index)
			{
				distancesToOtherClusters[clust] = (float)0;
			}
			else
			{
				float x, y, tempDistance;
				x = (clusters[index].getX() - clusters[clust].getX())*(clusters[index].getX() - clusters[clust].getX());
				y = (clusters[index].getY() - clusters[clust].getY())*(clusters[index].getY() - clusters[clust].getY());
				tempDistance = sqrt(x + y);
				distancesToOtherClusters[clust] = tempDistance;
			}
		}
	};

	float getX() 
	{
		return x; 
	}

	float getY() 
	{ 
		return y; 
	}

	float getDiameter() 
	{ 
		return diameter; 
	}

	int getIndex() 
	{ 
		return index; 
	}

	bool getChanged() 
	{ 
		return changed; 
	}

	float* getDistancesToOtherClusters() 
	{ 
		return distancesToOtherClusters;
	}

	void setChanged(bool flag) 
	{ 
		this->changed = flag; 
	}

	void setDiameter(float diameter) 
	{ 
		this->diameter = diameter; 
	}
};


#endif //__CLUSTER_H
