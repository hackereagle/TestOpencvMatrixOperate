#include "pch.h"

TEST(TestOpencvMat, TestArrayToMatWhetherSameMem)
{
	// ARRANGE
	char data[6] = { 1, 2, 3, 4, 5, 6 };

	// ACT
	cv::Mat mat = cv::Mat(2, 3, CV_8UC1, data);
	*(mat.data + 2) = 10;

	std::cout << "data = " << std::endl;
	PrintArray(data, 6);
	std::cout << "mat = \n" << mat << std::endl;

	// ASSERT
	for (int i = 0; i < 6; i++) {
		EXPECT_EQ(data[i], mat.data[i]);
	}
}