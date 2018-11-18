#ifndef __CLUSTER_H
#define __CLUSTER_H

#include <math.h>
#include <vector>
using namespace std;

//static const int INDEX_STARTER = 0;
//static int indexGen = INDEX_STARTER;
class Cluster
{
private:
	//static int indexGen;
	int index;
	bool changed; // i'm not sure this member is needed. this condition is tested directly in the Main.
	float x;
	float y;
	float diameter;
	int numberOfClusters;
	float* distancesToOtherClusters;
	
public:
	//static int indexGen;

	Cluster() { changed = true; /*index = indexGen++;*/ }
	void setCluster(int index, float x, float y, int numberOfClusters)
	{
		this->index = index;
		this->x = x;
		this->y = y;
		this->numberOfClusters = numberOfClusters;
		distancesToOtherClusters = new float[numberOfClusters];
	};
	~Cluster() { delete[] distancesToOtherClusters; }

	float getDistanceFromPoint(Point& p)
	{
		float pX = p.getX();
		float pY = p.getY();
		double defX = (x - pX)*(x - pX);
		double defY = (y - pY)*(y - pY);

		return (float)sqrt(defX + defY);
	};

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
				x = (float)(clusters[index].getX() - clusters[clust].getX())*(clusters[index].getX() - clusters[clust].getX());
				y = (float)(clusters[index].getY() - clusters[clust].getY())*(clusters[index].getY() - clusters[clust].getY());
				tempDistance = sqrt(x + y);
				distancesToOtherClusters[clust] = tempDistance;
			}
		}
	};

	float getX() { return x; }
	float getY() { return y; }
	float getDiameter() { return diameter; }
	bool getChanged() { return changed; }
	float* getDistancesToOtherClusters() { return distancesToOtherClusters;}
	void setChanged(bool flag) { this->changed = flag; }
	void setDiameter(float diameter) { this->diameter = diameter; }
//	static void initIndexGen() {indexGen = 0;}
	void print() const { cout << "(" << index << ") " << x << "\t" << y << endl; };
};


#endif //__CLUSTER_H
