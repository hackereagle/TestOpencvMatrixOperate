#pragma once

#include <iostream>
#include "opencv2/opencv.hpp"

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

inline cv::Mat PseudoInverseMat(cv::Mat mat)
{
	cv::Mat res;

	int m = mat.rows, n = mat.cols;

	//if (m == n) {
	//	std::cout << "in PseudoInverseMat, square matrix" << std::endl;
	//	res = mat.t() * mat;
	//	//res = mat * mat.t();
	//	std::cout << "in PseudoInverseMat, res = (mat^T * mat)\n" << res << std::endl;
	//	res = res.inv();
	//	std::cout << "in PseudoInverseMat, res = res^-1\n" << res << std::endl;
	//}
	//else {
	//	throw "Not implement";
	//}
	double ret = cv::invert(mat, res, cv::DecompTypes::DECOMP_SVD);
	std::cout << "in PseudoInverseMat, ret = " << ret << ", \nres = \n" << res << std::endl;

	return res;
}