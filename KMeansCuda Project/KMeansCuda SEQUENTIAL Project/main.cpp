
#include <fstream>
#include <time.h>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include "Point.h"
#include "Cluster.h"
#include "kernel.h"

//void writeNewFile(string fileName, Point* allPoints)
//{
//	ofstream outFile(fileName, ios::trunc);
//	outFile.clear();
//	for (int i = 0; i < 250000; i++)
//		outFile <<  i << "," << allPoints[i].getX() << "," << allPoints[i].getY()<< endl;
//}

void readFirstLine(ifstream& inFile, int& numOfPoints, int& maxNumOfClusters, int& maxIterations, int& qualityMeasure)
{
	char delimiterVal;

	inFile >> numOfPoints >> delimiterVal >> maxNumOfClusters >> delimiterVal >> maxIterations >> delimiterVal >> qualityMeasure;
}

void readPoint(ifstream& inFile, Point& point, int numOfClusters)
{
	int index;
	float x;
	float y;
	char delimiterVal;

	inFile >> index >> delimiterVal >> x >> delimiterVal >> y;

	point.setPoint(index, x, y, numOfClusters);
}

void writeToFile(string fileName, Cluster* currentClusters, int numOfClusters, int qualityMeasure)
{
	ofstream outFile(fileName, ios::trunc);
	outFile.clear();
	outFile << "Number of clusters with the best measure:" <<  endl;
	outFile << "K = " << numOfClusters << "\tQM = " << qualityMeasure <<  endl;
	outFile << "Centers of the clusters:" <<  endl;
	for (int i = 0; i < numOfClusters; i++)
		outFile << "(" << i << ") " << currentClusters[i].getX() << "\t" << currentClusters[i].getY() << endl;
}

void changeClusterCenter(Cluster* currentClusters, Point* allPoints, int numOfClusters, int numOfPoints)
{
	int i;
//#pragma omp parallel for shared(currentClusters, allPoints) private(i)
	for (i = 0; i < numOfClusters; i++)
	{
		float avgX=0, avgY=0;
		int num=0;
		for (int j = 0; j < numOfPoints; j++)
		{
			if (allPoints[j].getNearestCluster() == i)
			{
				avgX += allPoints[j].getX();
				avgY += allPoints[j].getY();
				num++;
			}
		}
		avgX /= num;
		avgY /= num;

		if (currentClusters[i].getX() != avgX  ||  currentClusters[i].getY() != avgY)
		{
			currentClusters[i].setCluster(i, avgX, avgY, numOfClusters);
			currentClusters[i].setChanged(true);
		}
		else
		{
			currentClusters[i].setChanged(false);
		}
	}
}

float getBestDistance(Cluster* clasters, int size)
{
	float distance = 0;

	for (int i = 0; i < size - 1; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			float x, y, tempDistance;
			x = (float)(pow((double)(clasters[i].getX() - clasters[j].getX()), 2));
			y = (float)(pow((double)(clasters[i].getY() - clasters[j].getY()), 2));
			tempDistance = sqrt(x + y);
			if ((i == 0 && j == 1) || (tempDistance < distance))
				distance = tempDistance;
		}
	}
	return distance;
}

void setBestClusters(Cluster* currentClusters, Cluster* bestClusters, int size)
{
	int i;
#pragma omp parallel for shared(bestClusters, currentClusters) private(i) 
	for (i = 0; i < size; i++)
		bestClusters[i] = currentClusters[i];
}

float checkQM(Cluster* currentClusters, int numOfClusters)
{
	float diamter, len, qm=0;
	for (int clust = 0; clust < numOfClusters; clust++)
	{
		diamter = currentClusters[clust].getDiameter();
		for (int i = 0; i<numOfClusters; i++)
		{
			len = currentClusters[clust].getDistancesToOtherClusters()[i];
			if ((int)len != 0)
				qm += diamter/len;
		}
	}

	return qm;
}

// HEAVY FUNCTION THAT TAKES A LONG TIME. PARALLELIZE?
void computeDiameters(Cluster* currentClusters, Point* allPoints,int numOfClusters, int numOfPoints)
{
	for (int clust = 0; clust < numOfClusters; clust++) // THIS FOR CAN BE PARALLELIZED with pragma.
	{
		float diameter = 0;
		for (int j = 0; j < numOfPoints - 1; j++)
		{
			for (int k = j + 1; k< numOfPoints; k++)
			{
				if (allPoints[j].getNearestCluster() == clust && allPoints[k].getNearestCluster() == clust)
				{
					float x, y, tempDistance;
					x = (float)(allPoints[j].getX() - allPoints[k].getX())*(allPoints[j].getX() - allPoints[k].getX());
					y = (float)(allPoints[j].getY() - allPoints[k].getY())*(allPoints[j].getY() - allPoints[k].getY());
					tempDistance = sqrt(x + y);
					if ((j == 0 && k == 1) || (tempDistance > diameter))
						diameter = tempDistance;
				}
			}
		}
		currentClusters[clust].setDiameter(diameter);
	}
}
using namespace std;

int main(int argc, char *argv[])
{
	
	const string IN_FILE = "C:\\Users\\afeka.ACADEMIC\\Desktop\\KMeansCuda SEQUENTIAL Project\\points10000.txt";
	const string OUT_FILE = "C:\\Users\\afeka.ACADEMIC\\Desktop\\KMeansCuda SEQUENTIAL Project\\results.txt";
	const int MASTER = 0;
	const int STARTING_K = 2;
	const int NUM_OF_MEMBERS_IN_POINT = 3;

	time_t startTime, endTime;
	Point* allPoints;
	Cluster* currentClusters;
	int numOfPoints; // N
	int maxNumOfClusters; //MAX
	int numOfClusters; //K
	int maxIterations; //LIMIT
	int qualityMeasure; // QM
	float bestDistance;
	int chunkSize;


	//MPI INITIALIZE
	int numprocs, myid, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Status status;

	initDevice();
	if (myid == MASTER) //MASTER
	{
		startTime = time(0);
		ifstream inFile(IN_FILE);
		readFirstLine(inFile, numOfPoints, maxNumOfClusters, maxIterations, qualityMeasure);
		cout << "numOfPoints = " << numOfPoints << ", maxNumOfClusters = " << maxNumOfClusters << ", maxIterations = " << maxIterations << ", qualityMeasure = " << qualityMeasure << endl;
		numOfClusters = STARTING_K;
		allPoints = new Point[numOfPoints];

		//reading points from file.
		for (int i = 0; i < numOfPoints; i++)
		{
			readPoint(inFile, allPoints[i], numOfClusters);
		}
		inFile.close(); //close file.
		cout << "Finished loading points from file." << endl;
		
		while (numOfClusters <= maxNumOfClusters)
		{
			// initiating a new array of clusters
//			Cluster::initIndexGen();
			for (int i = 0; i<numOfPoints; i++) // notifying points about new numOfClusters.
				allPoints[i].setNumOfClusters(numOfClusters);
			currentClusters = new Cluster[numOfClusters];
			for (int i = 0; i<numOfClusters; i++)
			{
				currentClusters[i].setCluster(i, allPoints[i].getX(), allPoints[i].getY(), numOfClusters);
//				cout << "currentCluster set, i=" << i << "\tx=" << allPoints[i].getX() << "\ty=" <<allPoints[i].getY() << endl;
				currentClusters[i].print();
			}

			// for each point define center that is most close to the point
			for (int i = 0; i< maxIterations; i++) // THIS FOR CAN BE PARALLELIZED with pragma.
			{
				bool centerChanged = false;
				for (int k = 0; k < numOfPoints; k++)
				{	
					for (int j = 0; j < numOfClusters; j++)
					{
						float temp = currentClusters[j].getDistanceFromPoint(allPoints[k]);
						if (temp != allPoints[k].getDistanceFromClusterIndex(j))
						{
							centerChanged = true;
							allPoints[k].setDistanceFromClusters(temp, j);
						}
					}
					allPoints[k].setNearestCluster();
				}
				//change center of clusters.
				if (centerChanged)
				{
					changeClusterCenter(currentClusters, allPoints, numOfClusters, numOfPoints);
				}
				else // no center has been changed
					break;
			}
			
			///////////////
			cout << "currentClusters are:" << endl;
			for (int i = 0; i<numOfClusters; i++)
			{
				//cout << "(" << i << ") " << currentClusters[i].getX() << "\t" << currentClusters[i].getY() << endl;
				currentClusters[i].print();
			}
			/////////////


			// test Quality Measure
			computeDiameters(currentClusters, allPoints, numOfClusters, numOfPoints);
			for (int i = 0; i < numOfClusters; i++)// THIS FOR CAN BE PARALLELIZED with pragma.
			{
				currentClusters[i].setDistancesToOtherClusters(currentClusters);
			}
 			float qm = checkQM(currentClusters, numOfClusters);
			cout << "qm = " << qm << "\n" << endl;
			if (qualityMeasure <= qm)
			{
				break;
			}
				


			//for (int i = 0; i<10; i++) 
			//{
			//	float temp = allPoints[i].getNearestCluster();
			//	cout << "(" << allPoints[i].getIndex() << ") " << allPoints[i].getX() << " " << allPoints[i].getY() << "\tNearest Cluster Index = " << temp << endl;
			//}

			numOfClusters++;
//			delete [] currentClusters; // for some reason, deleting currentClusters each time causes an exception when numOfClusters=47
		}
	}
	endTime = time(0);
	cout << "Finished job in " << endTime-startTime << endl;
	writeToFile(OUT_FILE, currentClusters, numOfClusters, qualityMeasure);
	MPI_Finalize();
	return 0;
}

