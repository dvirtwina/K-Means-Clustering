
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

void writeToFile(string fileName, Cluster* currentClusters, int numOfClusters, int QM)
{
	ofstream outFile(fileName, ios::trunc);
	outFile.clear();
	outFile << "Number of clusters with the best measure:" <<  endl;
	outFile << "K = " << numOfClusters << "\tQM = " << QM <<  endl;
	outFile << "Centers of the clusters:" <<  endl;
	for (int i = 0; i < numOfClusters; i++)
		outFile << "(" << i << ") " << currentClusters[i].getX() << "\t" << currentClusters[i].getY() << endl;
}

// MAYBE I CAN FURTHER MORE PARALLELIZE THIS WITH MPI
void changeClusterCenter(Cluster* currentClusters, Point* allPoints, int numOfClusters, int numOfPoints)
{
	int i;
#pragma omp parallel for shared(currentClusters, allPoints) private(i) // 
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
			currentClusters[i].setChanged(true); // MAYBE NOT NEEDED
		}
		else
		{
			currentClusters[i].setChanged(false);// MAYBE NOT NEEDED
		}
	}
}

float checkQM(Cluster* currentClusters, int numOfClusters)
{
	float diameter, len, qm = 0;
	int clust;
//#pragma omp parallel for shared(qm) private(clust)//PARALLELIZING CHANGES RESULTS. DONT KNOW WHY... BUT IT DOESN'T REALLY NEEDED
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

void computeMembers(int* numOfMembers, Point* allPoints, int numOfPoints)
{
	for (int i = 0; i<numOfPoints; i++)
	{
		numOfMembers[allPoints[i].getNearestCluster()]++;
	}
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
					x = (allPoints[j].getX() - allPoints[k].getX())*(allPoints[j].getX() - allPoints[k].getX());
					y = (allPoints[j].getY() - allPoints[k].getY())*(allPoints[j].getY() - allPoints[k].getY());
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
	
	const string IN_FILE = "C:\\Users\\afeka.ACADEMIC\\Desktop\\KMeansCuda CUDA Project\\points10000.txt";
	const string OUT_FILE = "C:\\Users\\afeka.ACADEMIC\\Desktop\\KMeansCuda CUDA Project\\results.txt";
	const int MASTER = 0;
	const int STARTING_K = 2;
	const int NUM_OF_MEMBERS_IN_POINT = 4;
	const int NUM_OF_MEMBERS_IN_CLUSTER = 4;
	int PHAZE_1 = 1;
	int PHAZE_2 = 2;
	int PHAZE_3 = 3;

	time_t startTime, endTime;
	Point* allPoints;
//	Point* chunkOfPoints;
	Cluster* currentClusters;
	int numOfPoints; // N
	int maxNumOfClusters; //MAX
	int numOfClusters; //K
	int maxIterations; //LIMIT
	int qualityMeasure; // QM
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
//			cout << "NEW WHILE ITERATION" << endl;

			// for each point define cluster center that is closest to it.
			for (int i = 0; i< maxIterations; i++)
			{
				//send currentClusters to slaves
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

					//setting the points to send to the slaves.
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
					float* pointsToGet = new float[currentChunk * (NUM_OF_MEMBERS_IN_POINT+1)]; // +1 for nearestClusterIndex
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
				for (int c = 0; c< numOfClusters; c++)
				{
					if (currentClusters[i].getChanged() == true)
						centerChanged = true;
				}
				if (centerChanged)
				{
					
					for (int p = 1; p< numprocs; p++) // notify slaves that part 1 of the job is still being carried out.
						MPI_Send(&PHAZE_1, 1, MPI_INTEGER, p, 0, MPI_COMM_WORLD);
				}
				else // no center has been changed
				{
					for (int p = 1; p< numprocs; p++) // notify slaves to change to phaze 2.
						MPI_Send(&PHAZE_2, 1, MPI_INTEGER, p, 0, MPI_COMM_WORLD);	
					break;
				}
			}
			
			/////////////// not part of the algorithm. TO BE REMOVED.
			cout << "currentClusters are:" << endl;
			for (int i = 0; i<numOfClusters; i++)
			{
				currentClusters[i].print();
			}
			int* numOfMembersArr = new int [numOfClusters]();
			computeMembers(numOfMembersArr, allPoints, numOfPoints);
			for (int i = 0 ; i<numOfClusters; i++)
			{
				cout << "members[" << i << "] = " << numOfMembersArr[i] << endl;
			}

			//float* dist = new float[numOfPoints];
			//cout << "BEFORE CUDA PROCESS" << endl;
			//copyPointsToDevice(allPoints, numOfPoints);
			//cudaDistancesOfPoint(allPoints, dist, allPoints[0].getX(), allPoints[0].getY());
			//cout << "AFTER CUDA PROCESS" << endl;
			//for (int i = 0; i<numOfPoints; i++)
			//{
			//	cout << "Point[" << i << "] dist = " << dist[i] << endl;
			//}
			/////////////

			// compute clusters diameters
			float* dist = new float[numOfPoints];
			copyPointsToDevice(allPoints, numOfPoints);
			for (int clust = 0; clust < numOfClusters; clust++)
			{
				float clustDiameter = 0;
				for (int i = 0; i < numOfPoints; i++)
				{
					cudaDistancesOfPoint(allPoints, dist, allPoints[i].getX(), allPoints[i].getY());

					float diameter = 0;
					for (int j = 0; j<numOfPoints; j++)
					{
						if (i != j && allPoints[i].getNearestCluster() == clust && allPoints[j].getNearestCluster() == clust)
						{
							if ((i == 0 && j == 1) 
								|| (j == 0)
								|| (dist[j] > diameter))
								diameter = dist[j];
						}
					}

					if (diameter > clustDiameter)
						clustDiameter = diameter;
				}
				currentClusters[clust].setDiameter(clustDiameter);
			}


			// test Quality Measure
//			computeDiameters(currentClusters, allPoints, numOfClusters, numOfPoints);
			for (i = 0; i < numOfClusters; i++)// THIS FOR CAN BE PARALLELIZED with pragma.
			{
				currentClusters[i].setDistancesToOtherClusters(currentClusters);
			}
 			float qm = checkQM(currentClusters, numOfClusters);
			cout << "qm = " << qm << endl << endl;
			if (qualityMeasure > qm)
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
		writeToFile(OUT_FILE, currentClusters, numOfClusters, qualityMeasure);
		delete [] currentClusters;
		delete [] allPoints;
	}
	else // SLAVES
	{
		int phaze = 1;
		int idx;

		while (phaze == PHAZE_1 || PHAZE_2)
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
					float* pointsToSend = new float[chunkSize * (NUM_OF_MEMBERS_IN_POINT+1)]; // +1 for nearestClusterIndex
					int idx = 0;
					for (int j = myid * (numOfPoints/numprocs); j < myid * (numOfPoints/numprocs) + chunkSize; j++)
					{
						pointsToSend[idx++] = (float)allPoints[j].getIndex();
						pointsToSend[idx++] = allPoints[j].getX();
						pointsToSend[idx++] = allPoints[j].getY();
						pointsToSend[idx++] = (float)numOfClusters;
						pointsToSend[idx++] = (float)allPoints[j].getNearestCluster();
					}

					MPI_Send(pointsToSend, chunkSize * (NUM_OF_MEMBERS_IN_POINT+1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

					delete [] pointsToSend;
					delete [] currentClusters;
					delete [] allPoints;
					MPI_Recv(&phaze, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
				}
				break;

			// a phaze that computes diameters.
			case 2: 
				{


					MPI_Recv(&phaze, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
					cout << "SLAVES NOTIFIED, phaze = " << phaze << endl;
				}
				break;
			}


		}
	}

	MPI_Finalize();
	return 0;
}