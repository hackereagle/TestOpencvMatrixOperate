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

TEST(TestOpencvSolveEquations, TestDeriveFourPointInverseAffineMatrix)
{
	// ARRANGE
	double Tx = 10.0, Ty = 20.0, theta = 60.0, Sx = 1.0, Sy = 1.0, ShearX = 2.0, ShearY = 3.0;
	AffineMatrixMultipleOrder multipler = AffineMatrixMultipleOrder(Tx, Ty, theta, Sx, Sy, ShearX, ShearY);
	// This is goal matrix.
	cv::Mat affine = multipler.CalculateAffineMatrix(AffineMatrix::Translation, AffineMatrix::Rotation, AffineMatrix::Shear, AffineMatrix::Scale);
	double stagesData[4][3] =
	{ {20.0, 30.0, 1.0}, 
	  {100.0, 10.0, 1.0}, 
	  {5.0, 66.0, 1.0}, 
	  {132.0, 168.0, 1.0} };
	cv::Mat stages;
	for (int i = 0; i < 4; i++) {
		stages.push_back(cv::Mat(1, 3, CV_64FC1, stagesData[i]));
	}
	stages = stages.t();
	cv::Mat wafers;
	wafers = affine * stages;
	std::cout << "stage points = \n" << stages << "\nwafer points = \n" << wafers << std::endl;

	cv::Mat invertAffineGoal;
	double ret = cv::invert(affine, invertAffineGoal);
	std::cout << "this is our goal matrix = \n" << invertAffineGoal << std::endl << std::endl;

	// ACT
	cv::Mat invertAffine;
	cv::Mat meta;
	double ret2 = cv::invert(wafers, meta, cv::DecompTypes::DECOMP_SVD);
	invertAffine = stages * meta;
	std::cout << "invert affine matrix = \n" << invertAffine << std::endl;

	// ASSERT
	for (int i = 0; i < 9; i++) {
		//EXPECT_DOUBLE_EQ(*((double*)(void*)invertAffine.data + i), *((double*)(void*)invertAffineGoal.data + i));
		EXPECT_NEAR(*((double*)(void*)invertAffine.data + i), *((double*)(void*)invertAffineGoal.data + i), 0.00001);
	}
}

TEST(TestOpencvSolveEquations, TestDeriveEightPointInverseAffineMatrix)
{
	// ARRANGE
	double Tx = 10.0, Ty = 20.0, theta = 60.0, Sx = 1.0, Sy = 1.0, ShearX = 2.0, ShearY = 3.0;
	AffineMatrixMultipleOrder multipler = AffineMatrixMultipleOrder(Tx, Ty, theta, Sx, Sy, ShearX, ShearY);
	// This is goal matrix.
	cv::Mat affine = multipler.CalculateAffineMatrix(AffineMatrix::Translation, AffineMatrix::Rotation, AffineMatrix::Shear, AffineMatrix::Scale);
	double stagesData[8][3] =
	{ {20.0, 37.0, 1.0}, 
	  {130.0, 189.0, 1.0}, 
	  {5.0, 66.0, 1.0}, 
	  {15.0, 66.0, 1.0}, 
	  {58.0, 61.99, 1.0}, 
	  {542.98, 6.13, 1.0}, 
	  {5.0, 667.31, 1.0}, 
	  {132.0, 168.0, 1.0} };
	cv::Mat stages;
	for (int i = 0; i < 8; i++) {
		stages.push_back(cv::Mat(1, 3, CV_64FC1, stagesData[i]));
	}
	stages = stages.t();
	cv::Mat wafers;
	wafers = affine * stages;
	std::cout << "stage points = \n" << stages << "\nwafer points = \n" << wafers << std::endl;

	cv::Mat invertAffineGoal;
	double ret = cv::invert(affine, invertAffineGoal);
	std::cout << "this is our goal matrix = \n" << invertAffineGoal << std::endl << std::endl;

	// ACT
	cv::Mat invertAffine;
	cv::Mat meta;
	double ret2 = cv::invert(wafers, meta, cv::DecompTypes::DECOMP_SVD);
	invertAffine = stages * meta;
	std::cout << "invert affine matrix = \n" << invertAffine << std::endl;

	// ASSERT
	for (int i = 0; i < 9; i++) {
		//EXPECT_DOUBLE_EQ(*((double*)(void*)invertAffine.data + i), *((double*)(void*)invertAffineGoal.data + i));
		EXPECT_NEAR(*((double*)(void*)invertAffine.data + i), *((double*)(void*)invertAffineGoal.data + i), 0.00001);
	}
}

TEST(TestOpencvSolveEquations, TestDeriveEightPointInverseAffineMatrixWithOpencvGetAffine)
{
	// ARRANGE
	double Tx = 10.0, Ty = 20.0, theta = 60.0, Sx = 1.0, Sy = 1.0, ShearX = 2.0, ShearY = 3.0;
	AffineMatrixMultipleOrder multipler = AffineMatrixMultipleOrder(Tx, Ty, theta, Sx, Sy, ShearX, ShearY);
	// This is goal matrix.
	cv::Mat affine = multipler.CalculateAffineMatrix(AffineMatrix::Translation, AffineMatrix::Rotation, AffineMatrix::Shear, AffineMatrix::Scale);
	double stagesData[8][3] =
	{ {20.0, 37.0, 1.0}, 
	  {130.0, 189.0, 1.0}, 
	  {5.0, 66.0, 1.0}, 
	  {15.0, 66.0, 1.0}, 
	  {58.0, 61.99, 1.0}, 
	  {542.98, 6.13, 1.0}, 
	  {5.0, 667.31, 1.0}, 
	  {132.0, 168.0, 1.0} };
	cv::Mat stages;
	for (int i = 0; i < 8; i++) {
		stages.push_back(cv::Mat(1, 3, CV_64FC1, stagesData[i]));
	}
	stages = stages.t();
	cv::Mat wafers;
	wafers = affine * stages;
	std::cout << "stage points = \n" << stages << "\nwafer points = \n" << wafers << std::endl;

	cv::Mat invertAffineGoal;
	double ret = cv::invert(affine, invertAffineGoal);
	std::cout << "this is our goal matrix = \n" << invertAffineGoal << std::endl << std::endl;

	// ACT
	cv::Point2f w[8];
	for (int i = 0; i < 8; i++) {
		w[i].x = static_cast<float>(*((double*)(void*)wafers.data + i));
		w[i].y = static_cast<float>(*((double*)(void*)wafers.data + i + wafers.cols));
	}
	cv::Point2f s[8];
	for (int i = 0; i < 8; i++) {
		s[i].x = static_cast<float>(*((double*)(void*)stages.data + i));
		s[i].y = static_cast<float>(*((double*)(void*)stages.data + i + stages.cols));
	}
	cv::Mat invertAffine = cv::getAffineTransform(w, s);
	std::cout << "invert affine matrix = \n" << invertAffine << std::endl;

	// ASSERT
	//std::cout << "==================" << std::endl;
	for (int i = 0; i < 6; i++) {
		//std::cout << *((double*)(void*)invertAffine.data + i) << ", " << *((double*)(void*)invertAffineGoal.data + i) << std::endl;
		EXPECT_NEAR(*((double*)(void*)invertAffine.data + i), *((double*)(void*)invertAffineGoal.data + i), 0.00001);
	}
}

TEST(TestOpencvSolveEquations, TestDeriveEightPointAffineMatrixWithOpencvGetAffine)
{
	// ARRANGE
	double Tx = 10.0, Ty = 20.0, theta = 60.0, Sx = 1.0, Sy = 1.0, ShearX = 2.0, ShearY = 3.0;
	AffineMatrixMultipleOrder multipler = AffineMatrixMultipleOrder(Tx, Ty, theta, Sx, Sy, ShearX, ShearY);
	// This is goal matrix.
	cv::Mat affine = multipler.CalculateAffineMatrix(AffineMatrix::Translation, AffineMatrix::Rotation, AffineMatrix::Shear, AffineMatrix::Scale);
	double stagesData[8][3] =
	{ {20.0, 37.0, 1.0}, 
	  {130.0, 189.0, 1.0}, 
	  {5.0, 66.0, 1.0}, 
	  {15.0, 66.0, 1.0}, 
	  {58.0, 61.99, 1.0}, 
	  {542.98, 6.13, 1.0}, 
	  {5.0, 667.31, 1.0}, 
	  {132.0, 168.0, 1.0} };
	cv::Mat stages;
	for (int i = 0; i < 8; i++) {
		stages.push_back(cv::Mat(1, 3, CV_64FC1, stagesData[i]));
	}
	stages = stages.t();
	cv::Mat wafers;
	wafers = affine * stages;
	std::cout << "stage points = \n" << stages << "\nwafer points = \n" << wafers << std::endl;

	// ACT
	cv::Point2f w[8];
	for (int i = 0; i < 8; i++) {
		w[i].x = static_cast<float>(*((double*)(void*)wafers.data + i));
		w[i].y = static_cast<float>(*((double*)(void*)wafers.data + i + wafers.cols));
	}
	cv::Point2f s[8];
	for (int i = 0; i < 8; i++) {
		s[i].x = static_cast<float>(*((double*)(void*)stages.data + i));
		s[i].y = static_cast<float>(*((double*)(void*)stages.data + i + stages.cols));
	}
	cv::Mat Affine = cv::getAffineTransform(s, w);
	std::cout << "affine matrix = \n" << Affine << std::endl;

	// ASSERT
	for (int i = 0; i < 6; i++) {
		EXPECT_NEAR(*((double*)(void*)Affine.data + i), *((double*)(void*)affine.data + i), 0.00001);
	}

}
