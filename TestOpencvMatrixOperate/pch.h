//
// pch.h
//

#pragma once

#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include "MatrixOperateExtensions.hpp"
#include <iostream>

template<class T>
inline void PrintArray(T* arr, size_t len)
{
	std::cout << "[";
	for (int i = 0; i < len; i++) {
		std::cout << arr[i] << " ";
	}
	std::cout << "]" << std::endl;
}

inline void PrintArray(char* arr, size_t len)
{
	std::cout << "[";
	for (int i = 0; i < len; i++) {
		std::cout << (int)arr[i] << " ";
	}
	std::cout << "]" << std::endl;
}

template<class T>
inline void PrintArray(T* arr, size_t row, size_t col)
{
	std::cout << "[";
	for (int i = 0; i < row; i++) {
		std::cout << "[";
		for (int j = 0; j < col; j++) {
			std::cout << arr[i][j] << " ";
		}

		if (i == row - 1) {
			std::cout << "]";
		}
		else {
			std::cout << "];" << std::endl;
		}
	}
	std::cout << "]" << std::endl;
}