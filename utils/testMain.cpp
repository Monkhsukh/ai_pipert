#include <iostream>
#include "HungarianAlgorithm.h"

/// https://github.com/mcximing/hungarian-algorithm-cpp

int main(void)
{
	// please use "-std=c++11" for this initialization of vector.
	vector<vector<double>> costMatrix = {{10, 19, 8, 15, 0},
										 {10, 18, 7, 17, 0},
										 {13, 16, 9, 14, 0},
										 {12, 19, 8, 18, 0.1}};

	HungarianAlgorithm HungAlgo;
	vector<int> assignment;

	double cost = HungAlgo.Solve(costMatrix, assignment);

	for (unsigned int x = 0; x < costMatrix.size(); x++)
		std::cout << x << "," << assignment[x] << "\t"; // 0,0    1,2     2,3     3,4

	std::cout << "\ncost: " << cost << std::endl; // 31.1

	return 0;
}