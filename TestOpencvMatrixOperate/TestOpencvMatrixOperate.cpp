#include "pch.h"

TEST(TestOpencvMatrixOperate, TestFloatMatrixMultiple) 
{
	// ARRANGE
	float* arr1 = new float[6] {1, 2, 3, 4, 5, 6};
	float* arr2 = new float[6] {1, 2, 3, 4, 5, 6};
	float* arr3 = new float[4] {22, 28, 49, 64};
	// opencv matrix multiple only allow float type
	// please refer to https://stackoverflow.com/questions/58438557/opencv-simple-2d-matrix-multiplication-fails
	cv::Mat expected = cv::Mat(2, 2, CV_32FC1, arr3);

	// ACT
	cv::Mat a = cv::Mat(2, 3, CV_32FC1, arr1);
	std::cout << a << std::endl;
	cv::Mat b = cv::Mat(3, 2, CV_32FC1, arr2);
	std::cout << b << std::endl;
	cv::Mat c = a * b;
	std::cout << c << std::endl;

	// ASSERT
	size_t len = static_cast<size_t>(c.rows) * static_cast<size_t>(c.cols);
	float* c_ptr = (float*)((void*)c.data);
	float* expected_ptr = (float*)((void*)expected.data);
	for (int i = 0; i < len; i++)
		EXPECT_EQ(*(c_ptr + i), *(expected_ptr + i));
	delete[] arr1;
	delete[] arr2;
	delete[] arr3;
}

TEST(TestOpencvMatrixOperate, TestIntMatrixMultipleFail) 
{
	// ARRANGE
	int* arr1 = new int[6] {1, 2, 3, 4, 5, 6};
	int* arr2 = new int[6] {1, 2, 3, 4, 5, 6};
	int* arr3 = new int[4] {22, 28, 49, 64};
	cv::Mat expected = cv::Mat(2, 2, CV_32SC1, arr3);

	// ACT
	cv::Mat a = cv::Mat(2, 3, CV_32SC1, arr1);
	std::cout << a << std::endl;
	cv::Mat b = cv::Mat(3, 2, CV_32SC1, arr2);
	std::cout << b << std::endl;
	// opencv matrix multiple only allow float type
	// please refer to https://stackoverflow.com/questions/58438557/opencv-simple-2d-matrix-multiplication-fails
	EXPECT_ANY_THROW(cv::Mat c = a * b;);

	delete[] arr1;
	delete[] arr2;
	delete[] arr3;
}

TEST(TestOpencvMatrixOperate, TestMatrixMultipleWithErrorRowAndCol) 
{
	// ARRANGE
	float* arr1 = new float[6] {1, 2, 3, 4, 5, 6};
	float* arr2 = new float[6] {1, 2, 3};

	// ACT
	cv::Mat a = cv::Mat(2, 3, CV_32FC1, arr1);
	cv::Mat b = cv::Mat(3, 1, CV_32FC1, arr2);

	// ASSERT
	EXPECT_ANY_THROW(cv::Mat c = b * a;);
}

TEST(TestOpencvMatrixOperate, TestFloatMatrixMultipleWithDecimalPoint) 
{
	// ARRANGE
	float* arr1 = new float[6] {1.2, 2, 3, 4, 5, 6};
	float* arr2 = new float[6] {1.2, 2, 3, 4, 5, 6};
	float* arr3 = new float[4] {22.440001, 28.4, 49.799999, 64};
	// opencv matrix multiple only allow float type
	// please refer to https://stackoverflow.com/questions/58438557/opencv-simple-2d-matrix-multiplication-fails
	cv::Mat expected = cv::Mat(2, 2, CV_32FC1, arr3);

	// ACT
	cv::Mat a = cv::Mat(2, 3, CV_32FC1, arr1);
	std::cout << a << std::endl;
	cv::Mat b = cv::Mat(3, 2, CV_32FC1, arr2);
	std::cout << b << std::endl;
	cv::Mat c = a * b;
	std::cout << c << std::endl;

	// ASSERT
	size_t len = static_cast<size_t>(c.rows) * static_cast<size_t>(c.cols);
	float* c_ptr = (float*)((void*)c.data);
	float* expected_ptr = (float*)((void*)expected.data);
	for (int i = 0; i < len; i++)
		EXPECT_FLOAT_EQ(*(c_ptr + i), *(expected_ptr + i));
	delete[] arr1;
	delete[] arr2;
	delete[] arr3;
}

TEST(TestOpencvMatrixOperate, TestMatrixTranspose)
{
	// ARRANGE
	float* arr1 = new float[6] {1, 2, 3, 4, 5, 6};
	cv::Mat expected = cv::Mat(3, 2, CV_32FC1, arr1);

	// ACT
	cv::Mat a = cv::Mat(2, 3, CV_32FC1, arr1);
	std::cout << a << std::endl;
	cv::Mat result = a.t();
	std::cout << result << std::endl;

	// ASSERT
	EXPECT_EQ(result.rows, expected.rows);
	EXPECT_EQ(result.cols, expected.cols);
	size_t len = static_cast<size_t>(result.rows) * static_cast<size_t>(result.cols);
	float* result_ptr = (float*)((void*)result.data);
	float* expected_ptr = (float*)((void*)expected.data);
	for (int i = 0; i < len; i++) {
		EXPECT_FLOAT_EQ(*(result_ptr), *(expected_ptr));
	}
	delete[] arr1;
}

TEST(TestOpencvMatrixOperate, TestSquareMatrixInverse)
{
#if 0
	// I don't know why using float is going to error
	// 
	// a = 
	// [1, 2, 3;
	// 5, 5, 6;
	// 7, 8, 9]
	// determint of a = 6
	// 	result =
	// 	[1, 2, 3;
	// 5, 5, 6;
	// 7, 8, 9]
	// a* result =
	// 	[1, 0, 0;
	// 0, 1, 0;
	// 4.7683716e-07, 0, 1]
	// Expected equality of these values :
	// *(unitMatPtr + i)
	// 	Which is : 4.7683716e-07
	// 	* (expPtr + i)
	// 	Which is : 0
	// ARRANGE
	float* data1 = new float[9] {1, 2, 3, 5, 5, 6, 7, 8, 9};
	float* expectedData1 = new float[9] {1, 0, 0, 0, 1, 0, 0, 0, 1};
	cv::Mat expected = cv::Mat(3, 3, CV_32FC1,  expectedData1);

	// ACT
	cv::Mat a = cv::Mat(3, 3, CV_32FC1, data1);
	std::cout << "a = \n" << a << std::endl;
	double det = cv::determinant(a);
	std::cout << "determint of a = " << det << std::endl;
	EXPECT_NE(det, 0.0);
	cv::Mat result = a.inv(cv::DecompTypes::DECOMP_LU);
	std::cout << "result = \n" << result << std::endl;

	// ASSERT
	cv::Mat unitMat = result * a;
	std::cout << "a * result = \n" << unitMat << std::endl;
	size_t len = static_cast<size_t>(unitMat.rows) * static_cast<size_t>(unitMat.cols);
	float* unitMatPtr = (float*)((void*)unitMat.data);
	float* expPtr = (float*)((void*)expected.data);
	for (int i = 0; i < len; i++) {
		EXPECT_FLOAT_EQ(*(unitMatPtr + i), *(expPtr + i));
	}
	delete[] data1;
	delete[] expectedData1;
#else
	// ARRANGE
	double* data1 = new double[9] {1, 2, 3, 5, 5, 6, 7, 8, 9};
	double* expectedData1 = new double[9] {1, 0, 0, 0, 1, 0, 0, 0, 1};
	cv::Mat expected = cv::Mat(3, 3, CV_64FC1,  expectedData1);

	// ACT
	cv::Mat a = cv::Mat(3, 3, CV_64FC1, data1);
	std::cout << "a = \n" << a << std::endl;
	double det = cv::determinant(a);
	std::cout << "determint of a = " << det << std::endl;
	EXPECT_NE(det, 0.0);
	//cv::Mat result = a.inv();
	cv::Mat result;
	double ret = cv::invert(a, result);
	std::cout << "cv::invert return value = " << ret << std::endl;
	EXPECT_NE(ret, 0.0);
	std::cout << "result = \n" << result << std::endl;

	// ASSERT
	cv::Mat unitMat = result * a;
	std::cout << "a * result = \n" << unitMat << std::endl;
	size_t len = static_cast<size_t>(unitMat.rows) * static_cast<size_t>(unitMat.cols);
	double* unitMatPtr = (double*)((void*)unitMat.data);
	double* expPtr = (double*)((void*)expected.data);
	for (int i = 0; i < len; i++) {
		EXPECT_DOUBLE_EQ(*(unitMatPtr + i), *(expPtr + i));
	}
	delete[] data1;
	delete[] expectedData1;
#endif
}

TEST(TestOpencvMatrixOperate, TestDetZeroMatrixInverseFail)
{
	// ARRANGE
	double* data1 = new double[9] {1, 2, 3, 4, 5, 6, 7, 8, 9};

	// ACT
	cv::Mat a = cv::Mat(3, 3, CV_64FC1, data1);
	std::cout << "a = \n" << a << std::endl;
	double det = cv::determinant(a);
	std::cout << "determint of a = " << det << std::endl;
	EXPECT_EQ(det, 0.0);
	cv::Mat result;
	double ret = cv::invert(a, result);
	std::cout << "cv::invert return value = " << ret << std::endl;
	EXPECT_EQ(ret, 0.0);
	std::cout << "result = \n" << result << std::endl;

	// ASSERT
	cv::Mat unitMat = result * a;
	std::cout << "a * result = \n" << unitMat << std::endl;
	size_t len = static_cast<size_t>(unitMat.rows) * static_cast<size_t>(unitMat.cols);
	double* unitMatPtr = (double*)((void*)unitMat.data);
	for (int i = 0; i < len; i++) {
		EXPECT_DOUBLE_EQ(*(unitMatPtr + i), 0.0);
	}
	delete[] data1;
}
