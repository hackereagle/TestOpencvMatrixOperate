#include "pch.h"
#include "MatrixOperateExtensions.hpp"

#define AFFINE_ENUM_STR_TBL \
	ENUM_STR_TBL(Translation, "Translation") \
	ENUM_STR_TBL(Rotation,    "Rotation") \
	ENUM_STR_TBL(Scale,       "Scale") \
	ENUM_STR_TBL(Shear,       "Shear")

#define ENUM_STR_TBL(e, s) e,
enum AffineMatrix : int
{
	AFFINE_ENUM_STR_TBL
};
#undef ENUM_STR_TBL

#define ENUM_STR_TBL(e, s) s,
const char* AffineStr[] = { AFFINE_ENUM_STR_TBL };
#undef ENUM_STR_TBL

class AffineMatrixMultipleOrder
{
public:
	AffineMatrixMultipleOrder()
	{
		InitializeAllAffineMatrixes();
		this->AssignOrder();
	}

	AffineMatrixMultipleOrder(double tx, double ty, double theta, double scaleX, double scaleY, double shearX, double shearY)
	{
		InitializeAllAffineMatrixes();
		this->AssignOrder();
		this->SetTraslation(tx, ty);
		this->SetRotation(theta);
		this->SetScale(scaleX, scaleY);
		this->SetShear(shearX, shearY);
	}

	void SetTraslation(double tx, double ty)
	{
		_tx = tx;
		_ty = ty;
		_translation.at<double>(0, 2) = _tx;
		_translation.at<double>(1, 2) = _ty;
		std::cout << "translation matrix = \n" << _translation << std::endl; // debug
	}

	void SetRotation(double theta)
	{
		_theta = theta;
		double t = DEG2RAD(_theta);
		_rotation.at<double>(0, 0) = cos(t);
		_rotation.at<double>(0, 1) = sin(t);
		_rotation.at<double>(1, 0) = -sin(t);
		_rotation.at<double>(1, 1) = cos(t);
		std::cout << "rotation matrix = \n" << _rotation << std::endl; // debug
	}
	
	void SetScale(double scaleX, double scaleY)
	{
		_scaleX = scaleX;
		_scaleY = scaleY;
		_scale.at<double>(0, 0) = _scaleX;
		_scale.at<double>(1, 1) = _scaleY;
		std::cout << "scale matrix = \n" << _scale << std::endl; // debug
	}

	void SetShear(double shearX, double shearY)
	{
		_shearX = shearX;
		_shearY = shearY;
		double Phi = DEG2RAD(_shearX), Psi = DEG2RAD(_shearY); // Phi is angle with x and Psi is angle with y.
		_shear.at<double>(0, 1) = tan(Phi);
		_shear.at<double>(1, 0) = tan(Psi);
		std::cout << "shear matrix = \n" << _shear << std::endl; // debug
	}

	cv::Mat CalculateAffineMatrix(AffineMatrix order1, AffineMatrix order2, AffineMatrix order3, AffineMatrix order4)
	{
		if (IsThereSameOrder(order1, order2, order3, order4))
			return cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
		
		cv::Mat res = *this->_affines[(int)order1] * *this->_affines[(int)order2] * *this->_affines[(int)order3] * *this->_affines[(int)order4];

		std::cout << AffineStr[(int)order1] << " * " << AffineStr[(int)order2] << " * " << AffineStr[(int)order3] << " * " << AffineStr[(int)order4] << " = \n" 
				  << res << std::endl;

		return res;
	}

private:
	double _tx = 0.0, _ty = 0.0;
	double _theta = 0.0;
	double _scaleX = 1.0, _scaleY = 1.0;
	double _shearX = 0.0, _shearY = 0.0;
	cv::Mat _translation;
	cv::Mat _rotation;
	cv::Mat _scale;
	cv::Mat _shear;
	cv::Mat* _affines[4]{nullptr};

	void InitializeAllAffineMatrixes()
	{
		//std::cout << "begin initialize" << std::endl; // debug
		_translation = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
		_translation.at<double>(0, 0) = 1.0;
		_translation.at<double>(1, 1) = 1.0;
		_translation.at<double>(2, 2) = 1.0;
		//std::cout << "translation matrix = \n" << _translation << std::endl; // debug

		_rotation = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
		double theta = DEG2RAD(_theta);
		_rotation.at<double>(0, 0) = 1.0;
		_rotation.at<double>(1, 1) = 1.0;
		_rotation.at<double>(2, 2) = 1.0;
		//std::cout << "rotation matrix = \n" << _rotation << std::endl; // debug

		_scale = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
		_scale.at<double>(0, 0) = 1.0;
		_scale.at<double>(1, 1) = 1.0;
		_scale.at<double>(2, 2) = 1.0;
		//std::cout << "scale matrix = \n" << _scale << std::endl; // debug

		_shear = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
		double Phi = DEG2RAD(_shearX), Psi = DEG2RAD(_shearY); // Phi is angle with x and Psi is angle with y.
		_shear.at<double>(0, 0) = 1.0;
		_shear.at<double>(1, 1) = 1.0;
		_shear.at<double>(2, 2) = 1.0;
		//std::cout << "shear matrix = \n" << _shear << std::endl; // debug
		//std::cout << "end initialize" << std::endl;
	}

	void AssignOrder()
	{
		this->_affines[AffineMatrix::Translation] = &this->_translation;
		this->_affines[AffineMatrix::Rotation] = &this->_rotation;
		this->_affines[AffineMatrix::Scale] = &this->_scale;
		this->_affines[AffineMatrix::Shear] = &this->_shear;
	}

	bool IsThereSameOrder(AffineMatrix order1, AffineMatrix order2, AffineMatrix order3, AffineMatrix order4)
	{
		bool isThereSameOrder = false;
		unsigned int count[4]{0};
		count[(int)order1] = count[(int)order1] + 1;
		count[(int)order2] = count[(int)order2] + 1;
		count[(int)order3] = count[(int)order3] + 1;
		count[(int)order4] = count[(int)order4] + 1;

		for (int i = 0; i < 4; i++) {
			if (count[i] > 1) {
				isThereSameOrder = true;
				std::cout << "repeat " << AffineStr[i] << "! So reject calculate!" << std::endl;
			}
		}

		return isThereSameOrder;
	}
};

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
