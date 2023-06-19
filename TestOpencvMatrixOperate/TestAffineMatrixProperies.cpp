#include "pch.h"
#include "MatrixOperateExtensions.hpp"

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
		double theta = DEG2RAD(_theta);
		_rotation.at<double>(0, 0) = cos(theta);
		_rotation.at<double>(0, 1) = sin(theta);
		_rotation.at<double>(1, 0) = -sin(theta);
		_rotation.at<double>(1, 1) = cos(theta);
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

	enum AffineMatrix : int
	{
		Translation,
		Rotation,
		Scale,
		Shear,
	};

	cv::Mat CalculateAffineMatrix(AffineMatrix order1, AffineMatrix order2, AffineMatrix order3, AffineMatrix order4)
	{
		// TODO: check same order
		
		// TODO: calculate affine matrix

		// TODO: output message

		// TODO: return
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
		_translation = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
		_translation.at<double>(0, 0) = 1.0;
		_translation.at<double>(1, 1) = 1.0;
		_translation.at<double>(0, 2) = _tx;
		_translation.at<double>(1, 2) = _ty;
		_translation.at<double>(2, 2) = 1.0;
		std::cout << "translation matrix = \n" << _translation << std::endl; // debug

		_rotation = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
		double theta = DEG2RAD(_theta);
		_rotation.at<double>(0, 0) = cos(theta);
		_rotation.at<double>(0, 1) = sin(theta);
		_rotation.at<double>(1, 0) = -sin(theta);
		_rotation.at<double>(1, 1) = cos(theta);
		_rotation.at<double>(2, 2) = 1.0;
		std::cout << "rotation matrix = \n" << _rotation << std::endl; // debug

		_scale = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
		_scale.at<double>(0, 0) = _scaleX;
		_scale.at<double>(1, 1) = _scaleY;
		_scale.at<double>(2, 2) = 1.0;
		std::cout << "scale matrix = \n" << _scale << std::endl; // debug

		_shear = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
		double Phi = DEG2RAD(_shearX), Psi = DEG2RAD(_shearY); // Phi is angle with x and Psi is angle with y.
		_shear.at<double>(0, 0) = 1.0;
		_shear.at<double>(1, 1) = 1.0;
		_shear.at<double>(0, 1) = tan(Phi);
		_shear.at<double>(1, 0) = tan(Psi);
		_shear.at<double>(2, 2) = 1.0;
		std::cout << "shear matrix = \n" << _shear << std::endl; // debug
	}

	void AssignOrder()
	{
		this->_affines[AffineMatrix::Translation] = &this->_translation;
		this->_affines[AffineMatrix::Rotation] = &this->_rotation;
		this->_affines[AffineMatrix::Scale] = &this->_scale;
		this->_affines[AffineMatrix::Shear] = &this->_shear;
	}
};

TEST(TestAffineMatrixProperties, TestAffineMatrixCommutativeLaw)
{
	// ARRANGE
	cv::Mat translation = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
	cv::Mat rotation = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
	cv::Mat scale = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
	cv::Mat shear = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
	// traslation
	double Tx = 10.0, Ty = 20.0;
	translation.at<double>(0, 0) = 1.0;
	translation.at<double>(1, 1) = 1.0;
	translation.at<double>(0, 2) = Tx;
	translation.at<double>(1, 2) = Ty;
	translation.at<double>(2, 2) = 1.0;
	std::cout << "translation matrix = \n" << translation << std::endl;
	// rotation
	double theta = DEG2RAD(60.0);
	rotation.at<double>(0, 0) = cos(theta);
	rotation.at<double>(0, 1) = sin(theta);
	rotation.at<double>(1, 0) = -sin(theta);
	rotation.at<double>(1, 1) = cos(theta);
	rotation.at<double>(2, 2) = 1.0;
	std::cout << "rotation matrix = \n" << rotation << std::endl;
	// scale
	//double Sx = 2.0, Sy = 3.0;
	double Sx = 1.0, Sy = 1.0;
	scale.at<double>(0, 0) = Sx;
	scale.at<double>(1, 1) = Sy;
	scale.at<double>(2, 2) = 1.0;
	std::cout << "scale matrix = \n" << scale << std::endl;
	// shear
	//double Phi = DEG2RAD(30.0), Psi = DEG2RAD(60.0); // Phi is angle with x and Psi is angle with y.
	double Phi = DEG2RAD(00.0), Psi = DEG2RAD(00.0); // Phi is angle with x and Psi is angle with y.
	shear.at<double>(0, 0) = 1.0;
	shear.at<double>(1, 1) = 1.0;
	shear.at<double>(0, 1) = tan(Phi);
	shear.at<double>(1, 0) = tan(Psi);
	shear.at<double>(2, 2) = 1.0;
	std::cout << "shear matrix = \n" << shear << std::endl;

	// ACT
	cv::Mat order1 = translation * rotation * shear * scale;
	std::cout << "order1 = translation * rotation * shear * scale\n" << order1 << std::endl;

	//cv::Mat order2 = translation * scale * rotation * shear;
	//std::cout << "order2 = translation * scale * rotation * shear\n" << order2 << std::endl;

	//cv::Mat order2 = scale * translation * rotation * shear;
	//std::cout << "order2 = scale * translation * rotation * shear\n" << order2 << std::endl;

	cv::Mat order2 = rotation * scale * translation * shear;
	std::cout << "order2 = rotation * scale * translation * shear\n" << order2 << std::endl;

	// ASSERT
	for (int i = 0; i < 6; i++) {
		EXPECT_DOUBLE_EQ(*((double*)(void*)order1.data + i), *((double*)(void*)order2.data + i));
	}
}
