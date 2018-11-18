// AUTHOR: Dvir Twina
#include <fstream>
#include <time.h>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include "Point.h"
#include "Cluster.h"
#include "kernel.h"

/** Read first line of the input file which contains instantiating data.*/
void readFirstLine(ifstream& inFile, int& numOfPoints, int& maxNumOfClusters, int& maxIterations, float& QM)
{
	char delimiterVal; // this variable is intended to be used if the input file contains delimiters.

	inFile >> numOfPoints /*>> delimiterVal*/ >> maxNumOfClusters /*>> delimiterVal*/ >> maxIterations /*>> delimiterVal*/ >> QM;
}

/** Read a line in the input file containing  the data of a single point.*/
void readPoint(ifstream& inFile, Point& point, int numOfClusters)
{
	int index;
	float x;
	float y;
	char delimiterVal; // this variable is intended to be used if the input file contains delimiters.

	inFile >> index /*>> delimiterVal*/ >> x /*>> delimiterVal*/ >> y;
	point.setPoint(index, x, y, numOfClusters);
}

/** Write final results to output file.*/
void writeToFile(string fileName, Cluster* currentClusters, int numOfClusters, float quailtyMeasure)
{
	ofstream outFile(fileName, ios::trunc);
	outFile.clear();
	outFile << "------------Final Results-------------" <<  endl;
	outFile << "K = " << numOfClusters << "\tQM = " << quailtyMeasure <<  endl;
	outFile << "Centers of the clusters:" <<  endl;
	for (int i = 0; i < numOfClusters; i++)
		outFile << "(" << i << ") " << currentClusters[i].getX() << "\t" << currentClusters[i].getY() << endl;
}

/** This function decides for each cluster its new center based on the average position of its points.*/
void changeClusterCenter(Cluster* currentClusters, Point* allPoints, int numOfClusters, int numOfPoints)
{
	int i;
#pragma omp parallel for shared(currentClusters, allPoints) private(i) // 
	for (i = 0; i < numOfClusters; i++)
	{
		float avgX = 0, avgY = 0;
		int members = 0;
		for (int j = 0; j < numOfPoints; j++)
		{
			if (allPoints[j].getClusterIndex() == i)
			{
				avgX += allPoints[j].getX();
				avgY += allPoints[j].getY();
				members++;
			}
		}
		avgX /= members;
		avgY /= members;

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

/** This function generates a measure based on a given set of clusters.*/
float checkQM(Cluster* currentClusters, int numOfClusters)
{
	float diameter, len, qm = 0;
	int clust;
	for (clust = 0; clust < numOfClusters; clust++)
	{
		diameter = currentClusters[clust].getDiameter();
		for (int i = 0; i<numOfClusters; i++)
		{
			len = currentClusters[clust].getDistancesToOtherClusters()[i];
			if ((int)len != 0)
				qm += diameter/len;
		}
	}

	return qm;
}

/** Computes the number of members (points) belonging to each cluster.*/
void computeMembers(int* numOfMembers, Point* allPoints, int numOfPoints)
{
	int i;
#pragma omp parallel for shared(allPoints) private(i)
	for (i = 0; i<numOfPoints; i++)
	{
		numOfMembers[allPoints[i].getClusterIndex()]++;
	}
}

using namespace std;

int main(int argc, char *argv[])
{
	
	const string IN_FILE = "C:\\Users\\afeka.ACADEMIC\\Desktop\\KMeansCuda Project\\cluster1.txt";
	const string OUT_FILE = "C:\\Users\\afeka.ACADEMIC\\Desktop\\KMeansCuda Project\\results.txt";
	const int MASTER = 0;
	const int STARTING_K = 2;
	const int NUM_OF_MEMBERS_IN_POINT = 4;
	const int NUM_OF_MEMBERS_IN_CLUSTER = 4;
	int PHAZE_1 = 1; // a phaze which defines a cluster for each point.
	int PHAZE_2 = 2; // a phaze that checks whether defining clusters is done.
	int PHAZE_3 = 3; // a phaze to terminate slaves.

	time_t startTime, endTime;
	Point* allPoints;
	Cluster* currentClusters;
	int numOfPoints;		// N
	int maxNumOfClusters;	// MAX
	int numOfClusters;		// K
	int maxIterations;		// LIMIT
	float QM;				// QM
	int chunkSize;
	float qm;


	// MPI INITIALIZE
	int numprocs, myid, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Status status;

	initDevice();
	if (myid == MASTER) // MASTER
	{
		startTime = time(0);
		ifstream inFile(IN_FILE);
		readFirstLine(inFile, numOfPoints, maxNumOfClusters, maxIterations, QM);
		cout << "numOfPoints = " << numOfPoints << ", maxNumOfClusters = " << maxNumOfClusters << ", maxIterations = " << maxIterations << ", QM = " << QM << endl;
		numOfClusters = STARTING_K;
		allPoints = new Point[numOfPoints];

		// reading points from file.
		for (int i = 0; i < numOfPoints; i++)
		{
			readPoint(inFile, allPoints[i], numOfClusters);
		}
		inFile.close();
		cout << "Finished loading points from file." << endl;
		
		while (numOfClusters <= maxNumOfClusters)
		{
			// initiating a new array of clusters
			int i;
#pragma omp parallel for shared(allPoints) private(i)
			for (i = 0; i<numOfPoints; i++) // notifying points about new numOfClusters.
				allPoints[i].setNumOfClusters(numOfClusters);

			currentClusters = new Cluster[numOfClusters];
#pragma omp parallel for shared(currentClusters, allPoints) private(i)
			for (i = 0; i<numOfClusters; i++)
			{
				currentClusters[i].setCluster(i, allPoints[i].getX(), allPoints[i].getY(), numOfClusters);
			}

			// for each point define cluster center that is closest to it.
			for (int i = 0; i< maxIterations; i++)
			{
				// send currentClusters to slaves
				for (int p = 1; p<numprocs; p++)
				{
					float* clustersToSend = new float[numOfClusters * NUM_OF_MEMBERS_IN_CLUSTER];
					int idx = 0;
					for (int c = 0; c<numOfClusters; c++)
					{
						clustersToSend[idx++] = (float)currentClusters[c].getIndex();
						clustersToSend[idx++] = currentClusters[c].getX();
						clustersToSend[idx++] = currentClusters[c].getY();
						clustersToSend[idx++] = (float)numOfClusters;
					}

					MPI_Send(&numOfClusters, 1, MPI_INTEGER, p, 0, MPI_COMM_WORLD);
					MPI_Send(clustersToSend, numOfClusters * NUM_OF_MEMBERS_IN_CLUSTER, MPI_FLOAT, p, 0, MPI_COMM_WORLD);

					delete [] clustersToSend;
				}

				// send allPoints and chunkSize(the workload) to slaves
				for (int p = 1; p<numprocs; p++)
				{
					chunkSize = numOfPoints/numprocs;
					if (numOfPoints%numprocs != 0 && p == numprocs-1)
						chunkSize = numOfPoints/numprocs + numOfPoints%numprocs;

					// setting the points to send to the slaves.
					float* pointsToSend = new float[numOfPoints * NUM_OF_MEMBERS_IN_POINT];
					int idx = 0;
					for (int j = 0; j < numOfPoints; j++)
					{
						pointsToSend[idx++] = (float)allPoints[j].getIndex();
						pointsToSend[idx++] = allPoints[j].getX();
						pointsToSend[idx++] = allPoints[j].getY();
						pointsToSend[idx++] = (float)numOfClusters;
					}

					MPI_Send(&numOfPoints, 1, MPI_INTEGER, p, 0, MPI_COMM_WORLD);
					MPI_Send(pointsToSend, numOfPoints * NUM_OF_MEMBERS_IN_POINT, MPI_FLOAT, p, 0, MPI_COMM_WORLD);

					MPI_Send(&chunkSize, 1, MPI_INTEGER, p, 0, MPI_COMM_WORLD);
					delete [] pointsToSend;
				}

				// define cluster of each point.
				chunkSize = numOfPoints/numprocs;
				int k;
#pragma omp parallel for shared(currentClusters, allPoints) private(k)
				for (k = myid * (numOfPoints/numprocs); k < chunkSize; k++)
				{	
					for (int j = 0; j < numOfClusters; j++)
					{
						float temp = currentClusters[j].getDistanceFromPoint(allPoints[k]);
						if (temp != allPoints[k].getDistanceFromClusterIndex(j))
						{
							allPoints[k].setDistanceFromClusters(temp, j);
						}
					}
					allPoints[k].setNearestCluster();
				}

				//get results from slave.
				for (int p = 1; p<numprocs; p++)
				{
					int currentChunk;
					if (p == numprocs-1)
						currentChunk = (numOfPoints/numprocs) + (numOfPoints%numprocs);
					else
						currentChunk = numOfPoints/numprocs;
					float* pointsToGet = new float[currentChunk * (NUM_OF_MEMBERS_IN_POINT+1)]; // +1 for clusterIndex
					MPI_Recv(pointsToGet, currentChunk * (NUM_OF_MEMBERS_IN_POINT+1), MPI_FLOAT, p, 0, MPI_COMM_WORLD, &status);

					int idx = 0;
					for (int i = p * (numOfPoints/numprocs); i<p * (numOfPoints/numprocs) + currentChunk; i++)
					{
						allPoints[i].setPoint((int)pointsToGet[idx], pointsToGet[idx+1], pointsToGet[idx+2], (int)pointsToGet[idx+3], (int)pointsToGet[idx+4]);
						idx += NUM_OF_MEMBERS_IN_POINT+1;
					}

					delete [] pointsToGet;
				}

				//change center of clusters.
				changeClusterCenter(currentClusters, allPoints, numOfClusters, numOfPoints);
				bool centerChanged = false;
				int c; 
#pragma omp parallel for shared(currentClusters) private(c)
				for (int c = 0; c < numOfClusters; c++)
				{
					if (currentClusters[c].getChanged() == true)
						centerChanged = true;
				}
				if (centerChanged)
				{
					
					for (int p = 1; p < numprocs; p++) // notify slaves that part 1 of the job is still being carried out.
						MPI_Send(&PHAZE_1, 1, MPI_INTEGER, p, 0, MPI_COMM_WORLD);
				}
				else // no center has been changed.
				{
					for (int p = 1; p < numprocs; p++) // notify slaves to change to phaze 2.
						MPI_Send(&PHAZE_2, 1, MPI_INTEGER, p, 0, MPI_COMM_WORLD);	
					break;
				}
			}

			// compute clusters diameters.
			int* numOfMembersArr = new int [numOfClusters]();
			computeMembers(numOfMembersArr, allPoints, numOfPoints);
			float* dist = new float[numOfPoints];
			copyPointsToDevice(allPoints, numOfPoints);
			for (int clust = 0; clust < numOfClusters; clust++)
			{
				// set a random point in the cluster.
				int pointIndex;
				int tempIdx;
				
				for (int i = 0; i<numOfPoints; i++)
				{
					if (allPoints[i].getClusterIndex() == clust)
					{
						pointIndex = i;
						break;
					}
				}

				// find diameter of the cluster.
				float maxDist = 0;
				for (int i = 0; i<3; i++)
				{
					cudaDistancesOfPoint(allPoints, dist, allPoints[pointIndex].getX(), allPoints[pointIndex].getY());
					bool cng = false;
					for (int j = 0, checker = 0; j<numOfPoints; j++)
					{
						if (pointIndex != j && allPoints[j].getClusterIndex() == clust)
						{
							if (dist[j] > maxDist)
							{
								maxDist = dist[j];
								tempIdx = j;
								cng = true;
								checker++;
							}
						}
						if (checker >= numOfMembersArr[clust])
							break;
					}

					if (cng)
						pointIndex = tempIdx;
				}
				currentClusters[clust].setDiameter(maxDist);
			}
			releaseDeviceMemory();

			// test Quality Measure
			for (i = 0; i < numOfClusters; i++)
			{
				currentClusters[i].setDistancesToOtherClusters(currentClusters);
			}
 			qm = checkQM(currentClusters, numOfClusters);
			if (QM < qm)
			{
				for (int p = 1; p< numprocs; p++) // notify slaves that part 1 of the job is still being carried out.
					MPI_Send(&PHAZE_1, 1, MPI_INTEGER, p, 0, MPI_COMM_WORLD);
			}
			else
			{
				for (int p = 1; p< numprocs; p++) // notify slaves job is done.
					MPI_Send(&PHAZE_3, 1, MPI_INTEGER, p, 0, MPI_COMM_WORLD);
				break;
			}

			numOfClusters++;
			delete [] currentClusters;
		}

		endTime = time(0);
		cout << "Finished job in " << endTime-startTime << endl;
		writeToFile(OUT_FILE, currentClusters, numOfClusters, qm);
		delete [] currentClusters;
		delete [] allPoints;
	}
	else // SLAVES
	{
		int phaze = 1;
		int idx;

		while (phaze == PHAZE_1 || phaze == PHAZE_2)
		{
			switch (phaze)
			{
			// a phaze which defines a cluster for each point.
			case 1: 
				{
					MPI_Recv(&numOfClusters, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
					float* clustersToGet = new float[numOfClusters * NUM_OF_MEMBERS_IN_CLUSTER];
					MPI_Recv(clustersToGet, numOfClusters * NUM_OF_MEMBERS_IN_CLUSTER, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
					currentClusters = new Cluster[numOfClusters];

					// construct currentClusters array
					idx = 0;
					for (int i = 0; i<numOfClusters; i++) 
					{
						currentClusters[i].setCluster((int)clustersToGet[idx], clustersToGet[idx+1], clustersToGet[idx+2], (int)clustersToGet[idx+3]);
						idx += NUM_OF_MEMBERS_IN_CLUSTER;
					}
					delete [] clustersToGet;

					MPI_Recv(&numOfPoints, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
					float* pointsToGet = new float[numOfPoints * NUM_OF_MEMBERS_IN_POINT];
					MPI_Recv(pointsToGet, numOfPoints * NUM_OF_MEMBERS_IN_POINT, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
					allPoints = new Point[numOfPoints];

					// construct allPoints array
					idx = 0;
					for (int i = 0; i<numOfPoints; i++) 
					{
						allPoints[i].setPoint((int)pointsToGet[idx], pointsToGet[idx+1], pointsToGet[idx+2], (int)pointsToGet[idx+3]);
						idx += NUM_OF_MEMBERS_IN_POINT;
					}
					delete [] pointsToGet;

					MPI_Recv(&chunkSize, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);

					// perform the actual job on the data.
					int k;
#pragma omp parallel for shared(currentClusters, allPoints) private(k)
					for (k = myid * (numOfPoints/numprocs); k < myid * (numOfPoints/numprocs) + chunkSize; k++)
					{	
						for (int j = 0; j < numOfClusters; j++)
						{
							float temp = currentClusters[j].getDistanceFromPoint(allPoints[k]);
							if (temp != allPoints[k].getDistanceFromClusterIndex(j))
							{
								allPoints[k].setDistanceFromClusters(temp, j);
							}
						}
						allPoints[k].setNearestCluster();
					}

					// send results to master
					float* pointsToSend = new float[chunkSize * (NUM_OF_MEMBERS_IN_POINT+1)]; // +1 for clusterIndex
					int idx = 0;
					for (int j = myid * (numOfPoints/numprocs); j < myid * (numOfPoints/numprocs) + chunkSize; j++)
					{
						pointsToSend[idx++] = (float)allPoints[j].getIndex();
						pointsToSend[idx++] = allPoints[j].getX();
						pointsToSend[idx++] = allPoints[j].getY();
						pointsToSend[idx++] = (float)numOfClusters;
						pointsToSend[idx++] = (float)allPoints[j].getClusterIndex();
					}

					MPI_Send(pointsToSend, chunkSize * (NUM_OF_MEMBERS_IN_POINT+1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

					delete [] pointsToSend;
					delete [] currentClusters;
					delete [] allPoints;
					MPI_Recv(&phaze, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
				}
				break;

			// this phaze only checks if MASTER keeps performing iterations to find cluster centers.
			case 2: 
				{
					MPI_Recv(&phaze, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
				}
				break;
			}
		}
	}

	MPI_Finalize();
	return 0;
}