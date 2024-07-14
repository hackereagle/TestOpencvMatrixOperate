#include "pch.h"
#include "MatrixOperateExtensions.hpp"

//TEST(TestAffineReflectionAndRotate, TestReflectAndRotateDiffMultOrder)
//{
//	// ARRANGE
//	double reflectionWithX[9]{1, 0, 0, 0, -1, 0, 0, 0, 1};
//}

void DoVectorAssert(cv::Mat& img2Stage)
{
	double vecInImg[3][3] = {
		{1, 0, 1},
		{0, 1, 1},
		{1, 1, 1},
	};
	double expectedResult[3][3] = {
		{1, 0, 1},
		{0, -1, 1},
		{1, -1, 1},
	};
	cv::Mat vecsInImg, vecsInStage;
	for (int i = 0; i < 3; i++) {
		vecsInImg.push_back(cv::Mat(1, 3, CV_64FC1, vecInImg[i]));
		vecsInStage.push_back(cv::Mat(1, 3, CV_64FC1, expectedResult[i]));
	}
	vecsInImg = vecsInImg.t();
	vecsInStage = vecsInStage.t();
	cv::Mat mat = img2Stage(cv::Rect(0, 0, img2Stage.cols, img2Stage.rows)).clone();
	*((double*)mat.data + 2) = 0;
	*((double*)mat.data + 5) = 0;

	// ACT
	cv::Mat result = mat * vecsInImg;
	std::cout << "result: " << result << std::endl;

	// ASSERT
	int len = vecsInImg.rows * vecsInImg.cols;
	for (int i = 0; i < len; i++) {
		EXPECT_NEAR(*((double*)result.data + i), *((double*)vecsInStage.data + i), 0.001);
	}
}

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
	DoVectorAssert(img2Stg);
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
	DoVectorAssert(img2Stg);
}