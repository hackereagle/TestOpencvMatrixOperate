#pragma once

#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"

#ifdef M_PI
#define PI M_PI
#elif PI
// do nothing
#else
#define PI (3.14159265358979323846)
#endif

#define DEG2RAD(x) ((x) * PI / 180.0)
#define RAD2DEG(x) ((x) * 180.0 / PI)

inline int Rank(cv::Mat mat)
{
	int rank = 0;
	try {
		// This is refering to https://stackoverflow.com/questions/37898019/how-to-calculate-matrix-rank-in-opencv
		// w is sigular matrix
		cv::Mat w, u, vt; 
		cv::SVD::compute(mat, w, u, vt);
		//std::cout << "mat = \n" << mat <<
		//			 "\nafter SVD, w = \n" << w <<
		//			 "u = \n" << u <<
		//			 "vt = \n" << vt << std::endl;

		cv::Mat noneZeroSigularValues = w > 0.001;
		//std::cout << "noneZeroSigularValues = \n" << noneZeroSigularValues << std::endl;

		rank = cv::countNonZero(noneZeroSigularValues);
	}
	catch (cv::Exception ex) {
		std::cout << "In Rank function, occur cv error: " << ex.what() << "\ninput = \n" << mat << std::endl;
	}
	catch (std::exception ex) {
		std::cout << "In Rank function, occur std error: " << ex.what() << "\ninput = \n" << mat << std::endl;
	}
	catch (...) {
		std::cout << "In Rank function, occur unexpected error" << std::endl;
	}

	return rank;
}

