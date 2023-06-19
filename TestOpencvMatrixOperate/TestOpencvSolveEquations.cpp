#include "pch.h"

TEST(TestOpencvSolveEquations, TestNonSquarePseudoInvereMatrixWithSvd)
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
	std::cout << "condition1 A * A^-1 * A result = \n" << cond1 << std::endl;
	for (int i = 0; i < len; i++)
		EXPECT_NEAR(*((double*)(void*)a.data + i), *((double*)(void*)cond1.data + i), 0.00000001);
}

TEST(TestOpencvSolveEquations, TestNonSquareLeftInverse)
{
	// ARRANGE
	double* data = new double[6] {1, 2, 3, 4, 5, 6};

	// ACT
	cv::Mat a = cv::Mat(2, 3, CV_64FC1, data);
	std::cout << a << std::endl;
	cv::Mat inv = a * a.t();
	double det = cv::determinant(inv);
	std::cout << "A * A^T = \n" << inv << "\ndet = " << det << std::endl;
	EXPECT_NE(det, 0.0);
	inv = a.t() * inv.inv();
	std::cout << "A^-1 = A^T * (A * A^T)^-1 = \n" << inv << std::endl;
	cv::Mat ag = a * inv;
	std::cout << "AG = \n" << ag << std::endl;

	// ASSERT
	size_t len = static_cast<size_t>(a.rows) * static_cast<size_t>(a.cols);
	cv::Mat cond1 = a * inv * a;
	std::cout << "condition1 A * A^-1 * A result = \n" << cond1 << std::endl;
	for (int i = 0; i < len; i++)
		EXPECT_NEAR(*((double*)(void*)a.data + i), *((double*)(void*)cond1.data + i), 0.00000001);
}


TEST(TestOpencvSolveEquations, TestNonSquareRightInverseMatrixWithSvd)
{	
	// ARRANGE
	double* data = new double[6] {1, 2, 3, 4, 5, 6};
	
	// ACT
	cv::Mat a = cv::Mat(3, 2, CV_64FC1, data);
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
	std::cout << "condition1 A * A^-1 * A result = \n" << cond1 << std::endl;
	for (int i = 0; i < len; i++)
		EXPECT_NEAR(*((double*)(void*)a.data + i), *((double*)(void*)cond1.data + i), 0.00000001);
}

TEST(TestOpencvSolveEquations, TestNonSquareRightInverse)
{
	// ARRANGE
	double* data = new double[6] {1, 2, 3, 4, 5, 6};

	// ACT
	cv::Mat a = cv::Mat(3, 2, CV_64FC1, data);
	std::cout << a << std::endl;
	cv::Mat inv = a.t() * a;
	double det = cv::determinant(inv);
	std::cout << "A^T * A = \n" << inv << "\ndet = " << det << std::endl;
	EXPECT_NE(det, 0.0);
	inv = inv.inv() * a.t();
	std::cout << "A^-1 = (A^T * A)^-1 * A^T = \n" << inv << std::endl;
	cv::Mat ag = a * inv;
	std::cout << "AG = \n" << ag << std::endl;

	// ASSERT
	size_t len = static_cast<size_t>(a.rows) * static_cast<size_t>(a.cols);
	cv::Mat cond1 = a * inv * a;
	std::cout << "condition1 A * A^-1 * A result = \n" << cond1 << std::endl;
	for (int i = 0; i < len; i++)
		EXPECT_NEAR(*((double*)(void*)a.data + i), *((double*)(void*)cond1.data + i), 0.00000001);
}
