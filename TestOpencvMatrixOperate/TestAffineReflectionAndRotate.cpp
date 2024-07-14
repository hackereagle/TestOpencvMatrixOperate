#include "pch.h"
#include "MatrixOperateExtensions.hpp"

//TEST(TestAffineMatrixProperties, TestReflectAndRotateDiffMultOrder)
//{
//	// ARRANGE
//	double reflectionWithX[9]{1, 0, 0, 0, -1, 0, 0, 0, 1};
//}

TEST(TestAffineReflectionAndRotate, TestSolveFromPoints)
{
	// ARRANGE
	double stagePoints[4][3] =
	{
		{100, 100, 1},
		{4000, 300, 1},
		{200, 4000, 1},
		{4100, 4300, 1},
	};
	double imagePoints[4][3] =
	{
		{100, 4400, 1},
		{4000, 4200, 1},
		{200, 500, 1},
		{4100, 200, 1},
	};
	cv::Mat stagePts, imagePts;
	for (int i = 0; i < 4; i++) {
		stagePts.push_back(cv::Mat(1, 3, CV_64FC1, stagePoints[i]));
		imagePts.push_back(cv::Mat(1, 3, CV_64FC1, imagePoints[i]));	
	}
	stagePts = stagePts.t();
	imagePts = imagePts.t();

	// ACT
	cv::Mat meta;
	cv::invert(imagePts, meta, cv::DecompTypes::DECOMP_SVD);
	cv::Mat img2Stg = stagePts * meta;
	std::cout << img2Stg << std::endl;

	// ASSERT
	EXPECT_NEAR(*((double*)img2Stg.data + 0),  1.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 1),  0.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 2),  0.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 3),  0.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 4), -1.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 5),  4500.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 6),  0.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 7),  0.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 8),  1.0, 0.001);
}

TEST(TestAffineReflectionAndRotate, TestSolveFromPoints_WithTranslate)
{
	// ARRANGE
	double stagePoints[4][3] =
	{
		{700, 600, 1},
		{4600, 800, 1},
		{800, 4500, 1},
		{4700, 4800, 1},
	};
	double imagePoints[4][3] =
	{
		{100, 4400, 1},
		{4000, 4200, 1},
		{200, 500, 1},
		{4100, 200, 1},
	};
	cv::Mat stagePts, imagePts;
	for (int i = 0; i < 4; i++) {
		stagePts.push_back(cv::Mat(1, 3, CV_64FC1, stagePoints[i]));
		imagePts.push_back(cv::Mat(1, 3, CV_64FC1, imagePoints[i]));	
	}
	stagePts = stagePts.t();
	imagePts = imagePts.t();

	// ACT
	cv::Mat meta;
	cv::invert(imagePts, meta, cv::DecompTypes::DECOMP_SVD);
	cv::Mat img2Stg = stagePts * meta;
	std::cout << img2Stg << std::endl;

	// ASSERT
	EXPECT_NEAR(*((double*)img2Stg.data + 0),  1.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 1),  0.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 2),  600.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 3),  0.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 4), -1.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 5),  5000.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 6),  0.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 7),  0.0, 0.001);
	EXPECT_NEAR(*((double*)img2Stg.data + 8),  1.0, 0.001);
}