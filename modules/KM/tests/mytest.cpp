#include "../include/pybind11/eigen.h"
#include "../include/pybind11/stl.h"

#if defined(_MSC_VER)
#  pragma warning(disable: 4996) // C4996: std::unary_negation is deprecated
#endif

#include <vector>


std::vector<std::vector<int>> plusOne (std::vector<std::vector<int>> a) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			a[i][j] ++;
		}
	}
	return a;
}


Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> myFunc(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> mat_a) {
	std::vector<std::vector<int> > vec_a (mat_a.rows(), std::vector<int> (mat_a.cols()));
	
	for (int i = 0; i < mat_a.rows(); i++) {
		for (int j = 0; j < mat_a.cols(); j++) {
			vec_a[i][j] = mat_a(i,j);
		}
	} 

	std::vector<std::vector<int>> vec_res = plusOne(vec_a);

	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> mat_res(mat_a.rows(), mat_a.cols());
	
	for (int i = 0; i < mat_a.rows(); i++) {
		for (int j = 0; j < mat_a.cols(); j++) {
			mat_res(i,j) = vec_res[i][j];
		}
	}

	return mat_res;
}


PYBIND11_MODULE(mytest, m) {
    m.def("myFunc", &myFunc);
}






