#include "pch.h"

TEST(TestOpencvMatrixOperate, TestNonSquarePseudoInvereMatrixWithSvd)
{
	// ARRANGE
	double* data = new double[6] {1, 2, 3, 4, 5, 6};
	
	// ACT
	cv::Mat a = cv::Mat(2, 3, CV_64FC1, data);
	std::cout << a << std::endl;
	cv::Mat inv;
	double ret = cv::invert(a, inv, cv::DecompTypes::DECOMP_SVD);
	EXPECT_NE(ret, 0.0);
	std::cout << "ret = " << ret << 
				 "\ninv = " << inv << std::endl;
	cv::Mat ag = a * inv;
	std::cout << "AG = \n" << ag << std::endl;
	

	// ASSERT
	size_t len = static_cast<size_t>(a.rows) * static_cast<size_t>(a.cols);
	cv::Mat cond1 = a * inv * a;
	std::cout << cond1 << std::endl;
	for (int i = 0; i < len; i++)
		EXPECT_NEAR(*((double*)(void*)a.data + i), *((double*)(void*)cond1.data + i), 0.00000001);
}