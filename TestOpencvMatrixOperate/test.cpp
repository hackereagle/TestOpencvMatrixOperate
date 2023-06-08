#include "pch.h"

TEST(TestOpencvMatrixOperate, TestMatrixMultiple) 
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
	cv::Mat c = a * b;

	//// ARRANGE
	//int arr1[2][3] = { {1, 2, 3}, {4, 5, 6 } };
	//int arr2[3][2] = { {1, 2}, {3, 4}, {5, 6} };
	//int arr3[2][2] = { {22, 28}, {49, 64 } };
	//cv::Mat expected = cv::Mat(2, 2, CV_32SC1, arr3);

	//// ACT
	//cv::Mat a = cv::Mat(2, 3, CV_32SC1, arr1);
	//std::cout << a << std::endl;
	//cv::Mat b = cv::Mat(3, 2, CV_32SC1, arr2);
	//std::cout << b << std::endl;
	//cv::Mat c = a* b;

	// ASSERT
	//EXPECT_EQ(c, expected);
	size_t len = c.rows * c.cols;
	int* c_ptr = (int*)((void*)c.data);
	int* expected_ptr = (int*)((void*)expected.data);
	for (int i = 0; i < len; i++)
		EXPECT_EQ(*(c_ptr + i), *(expected_ptr + i));
	delete[] arr1;
	delete[] arr2;
	delete[] arr3;
}

TEST(TestOpencvMatrixOperate, TestMatrixMultipleWithErrorRowAndCol) 
{
	// ARRANGE
	int* arr1 = new int[6] {1, 2, 3, 4, 5, 6};
	int* arr2 = new int[6] {1, 2, 3, 4, 5, 6};

	// ACT
	cv::Mat a = cv::Mat(2, 3, CV_32SC1, arr1);
	cv::Mat b = cv::Mat(3, 2, CV_32SC1, arr2);

	// ASSERT
	EXPECT_ANY_THROW(cv::Mat c = a.mul(b););
}
