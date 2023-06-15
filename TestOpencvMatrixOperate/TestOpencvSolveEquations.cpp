#include "pch.h"

TEST(TestOpencvMatrixOperate, TestNonSquarePseudoInvereMatrix)
{
	// ARRANGE
	double* data = new double[6] {1, 2, 3, 4, 5, 6};
	
	// ACT
	cv::Mat a = cv::Mat(2, 3, CV_64FC1, data);
	std::cout << a << std::endl;
	cv::Mat inv;
	//double ret = cv::invert(a, inv, cv::DecompTypes::DECOMP_LU);
	//cv::Mat tmp = a.t() * a;
	cv::Mat tmp = a.t() * a;
	double det = cv::determinant(tmp);
	std::cout << "determinat of tmp = " << det << std::endl;
	EXPECT_NE(det, 0.0);
	std::cout << "tmp = " << tmp << std::endl;
	double ret = cv::invert(tmp, inv, cv::DecompTypes::DECOMP_LU);
	EXPECT_NE(ret, 0.0);
	//inv = inv * a.t();
	std::cout << "ret = " << ret << 
				 "\ninv = " << inv << std::endl;
	

	// ASSERT
	size_t len = static_cast<size_t>(a.rows) * static_cast<size_t>(a.cols);
	cv::Mat cond1 = a * inv * a;
	std::cout << cond1 << std::endl;
	for (int i = 0; i < len; i++)
		EXPECT_DOUBLE_EQ(*((double*)(void*)a.data + i), *((double*)(void*)cond1.data + i));
}