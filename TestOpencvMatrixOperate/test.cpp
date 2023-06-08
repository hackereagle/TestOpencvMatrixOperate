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
	size_t len = c.rows * c.cols;
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
	size_t len = c.rows * c.cols;
	float* c_ptr = (float*)((void*)c.data);
	float* expected_ptr = (float*)((void*)expected.data);
	for (int i = 0; i < len; i++)
		EXPECT_FLOAT_EQ(*(c_ptr + i), *(expected_ptr + i));
	delete[] arr1;
	delete[] arr2;
	delete[] arr3;
}
