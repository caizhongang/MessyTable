#include "../include/pybind11/pybind11.h"
#include "../include/pybind11/eigen.h"
#include "../include/pybind11/stl.h"

#if defined(_MSC_VER)
#  pragma warning(disable: 4996) // C4996: std::unary_negation is deprecated
#endif

#include <vector>
#include "hungarian.hpp"

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> solve(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> costMatrix, int threshold) {
	int num_rows = costMatrix.rows();
	int num_cols = costMatrix.cols();

	// create a input matrix to km
	// note cost above threshold are considered invalid match right away
	// and they are treated to be "equally invalid", remaining the default value threshold+1
	// to avoid affect matching with the valid costs
	Hungarian::Matrix cost2DVec (num_rows, std::vector<int>(num_cols, threshold+1));
	for (int i = 0; i < num_rows; i++) {
		for (int j = 0; j < num_cols; j++) {
			if (costMatrix(i,j) <= threshold) {			
				cost2DVec[i][j] = costMatrix(i,j);
			}
		}
	}

	// solve KM
	Hungarian::Result res2DVec = Hungarian::Solve(cost2DVec, Hungarian::MODE_MINIMIZE_COST);
	if (!res2DVec.success) {
		std::cout << "Matching not successful. Return input matrix itself." << std::endl;
		return costMatrix;
	}

	// convert 2d std vector to eigen matrix
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> matchMatrix (num_rows, num_cols);
	for (int i = 0; i < num_rows; i++) {
		for (int j = 0; j < num_cols; j++) {
			if (res2DVec.assignment[i][j] == 1 && cost2DVec[i][j] <= threshold) {
				// only accept a match if
				// KM agorithm assigns a match and
				// the match can be valid in the first place (cost <= threshold)
				matchMatrix(i,j) = 1;
			} else {
				matchMatrix(i,j) = 0;
			}			
		}
	}

	return matchMatrix;
}

namespace py = pybind11;

PYBIND11_MODULE(KMSolver, m) {
    m.def("solve", &solve, py::arg("cost_matrix"), py::arg("threshold"));
}



