#include "pch.h"
#include "MatrixOperateExtensions.hpp"

TEST(TestAffineMatrixProperties, TestAffineMatrixNotCommutativeLawCase)
{
	// ARRANGE
	cv::Mat translation = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
	cv::Mat rotation = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
	cv::Mat scale = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
	cv::Mat shear = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));

	// ACT
	// theta, ShearX and ShearY are degree.
	double Tx = 10.0, Ty = 20.0, theta = 60.0, Sx = 1.0, Sy = 1.0, ShearX = 0.0, ShearY = 0.0;
	AffineMatrixMultipleOrder multipler = AffineMatrixMultipleOrder(Tx, Ty, theta, Sx, Sy, ShearX, ShearY);
	cv::Mat order1 = multipler.CalculateAffineMatrix(AffineMatrix::Translation, AffineMatrix::Rotation, AffineMatrix::Shear, AffineMatrix::Scale);
	cv::Mat order2 = multipler.CalculateAffineMatrix(AffineMatrix::Rotation, AffineMatrix::Scale, AffineMatrix::Translation, AffineMatrix::Shear);

	// ASSERT
	int count = 0;
	for (int i = 0; i < 9; i++) {
		::testing::internal::FloatingPoint<double> l(*((double*)(void*)order1.data + i)), r(*((double*)(void*)order2.data + i));
		if (l.AlmostEquals(r)) {
			count = count + 1;
		}
	}

	ASSERT_FALSE(count == 9);
}

TEST(TestAffineMatrixProperties, TranslationFirst_NoScaleAndShear_SwitchRotationAndScale)
{
	// ARRANGE
	cv::Mat translation = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
	cv::Mat rotation = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
	cv::Mat scale = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
	cv::Mat shear = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));

	// ACT
	// theta, ShearX and ShearY are degree.
	double Tx = 10.0, Ty = 20.0, theta = 60.0, Sx = 1.0, Sy = 1.0, ShearX = 0.0, ShearY = 0.0;
	AffineMatrixMultipleOrder multipler = AffineMatrixMultipleOrder(Tx, Ty, theta, Sx, Sy, ShearX, ShearY);
	cv::Mat order1 = multipler.CalculateAffineMatrix(AffineMatrix::Translation, AffineMatrix::Rotation, AffineMatrix::Scale, AffineMatrix::Shear);
	cv::Mat order2 = multipler.CalculateAffineMatrix(AffineMatrix::Translation, AffineMatrix::Scale, AffineMatrix::Rotation, AffineMatrix::Shear);

	// ASSERT
	int count = 0;
	for (int i = 0; i < 9; i++) {
		::testing::internal::FloatingPoint<double> l(*((double*)(void*)order1.data + i)), r(*((double*)(void*)order2.data + i));
		if (l.AlmostEquals(r)) {
			count = count + 1;
		}
	}

	ASSERT_TRUE(count == 9);
}
