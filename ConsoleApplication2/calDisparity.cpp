#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <string.h>
#include <tchar.h>
#include <iostream>
#include<opencv2/imgproc/types_c.h> 
#include<opencv2/opencv.hpp>
#include<algorithm> 

using namespace std;
using namespace cv;

//Lower the bit resolution
void change_bits(Mat imgL, Mat imgR, int bits)
{
	for (int i = 0; i < imgL.rows; i++)
	{
		for (int j = 0; j < imgL.cols; j++)
		{
		/*
		img.at<cv::Vec3b>(i, j)[0] = (int)(img.at<cv::Vec3b>(i, j)[0] / pow(2, bits)) * pow(2, bits);
		img.at<cv::Vec3b>(i, j)[1] = (int)(img.at<cv::Vec3b>(i, j)[1] / pow(2, bits)) * pow(2, bits);
		img.at<cv::Vec3b>(i, j)[2] = (int)(img.at<cv::Vec3b>(i, j)[2] / pow(2, bits)) * pow(2, bits);*/

		imgL.at<uchar>(i, j) = (int)(imgL.at<uchar>(i, j) / pow(2, bits)) * pow(2, bits);
		imgR.at<uchar>(i, j) = (int)(imgR.at<uchar>(i, j) / pow(2, bits)) * pow(2, bits);
		}
	}
}


//return a size*size Mat, size must be odd number
Mat get_template(Mat img, int anchor_y, int anchor_x, int size)
{
	Mat temp = Mat(size, size, CV_8UC1);
	if (anchor_x < size / 2 && anchor_y < size / 2) //anchor too close to left-up corner
	{
		for (int i = 0; i < anchor_y + size / 2 + 1; i++)
		{
			for (int j = 0; j < anchor_x + size / 2 + 1; j++)
			{
				temp.at<uchar>(i + size / 2 - anchor_y, j + size / 2 - anchor_x) = img.at<uchar>(i, j);
			}
		}
	}
	else if (anchor_x > img.cols - 1 - size / 2 && anchor_y < size / 2) //anchor too close to right-up corner
	{
		for (int i = 0; i < anchor_y + size / 2; i++)
		{
			for (int j = anchor_x - size / 2; j < img.cols; j++)
			{
				temp.at<uchar>(i + size / 2 - anchor_y, j - (anchor_x - size / 2)) = img.at<uchar>(i, j);
			}
		}
	}
	else if (anchor_x < size / 2 && anchor_y > img.rows - 1 - size / 2) //anchor too close to left-down corner
	{
		for (int i = anchor_y - size / 2; i < img.rows; i++)
		{
			for (int j = 0; j < anchor_x + size / 2; j++)
			{
				temp.at<uchar>(i - (anchor_y - size / 2), j + size / 2 - anchor_x) = img.at<uchar>(i, j);
			}
		}
	}
	else if (anchor_x > img.cols - 1 - size / 2 && anchor_y > img.rows - 1 - size / 2) //anchor too close to right-down corner
	{
		for (int i = anchor_y - size / 2; i < img.rows; i++)
		{
			for (int j = anchor_x - size / 2; j < img.cols; j++)
			{
				temp.at<uchar>(i - (anchor_y - size / 2), j - (anchor_x - size / 2)) = img.at<uchar>(i, j);
			}
		}
	}
	else if (anchor_x < size / 2) //anchor too close to left edge
	{
		for (int i = anchor_y - size / 2; i < anchor_y + size / 2 + 1; i++)
		{
			for (int j = 0; j < anchor_x + size / 2 + 1; j++)
			{
				temp.at<uchar>(i - (anchor_y - size / 2), j + size / 2 - anchor_x) = img.at<uchar>(i, j);
			}
		}
	}
	else if (anchor_y < size / 2) //anchor too close to upper edge
	{
		for (int i = 0; i < anchor_y + size / 2 + 1; i++)
		{
			for (int j = anchor_x - size / 2; j < anchor_x + size / 2 + 1; j++)
			{
				temp.at<uchar>(i + size / 2 - anchor_y, j - (anchor_x - size / 2)) = img.at<uchar>(i, j);
			}
		}
	}
	else if (anchor_x > img.cols - 1 - size / 2) //anchor too close to right edge
	{
		for (int i = anchor_y - size / 2; i < anchor_y + size / 2 + 1; i++)
		{
			for (int j = anchor_x - size / 2; j < img.cols; j++)
			{
				temp.at<uchar>(i - (anchor_y - size / 2), j - (anchor_x - size / 2)) = img.at<uchar>(i, j);
			}
		}
	}
	else if (anchor_y > img.rows - 1 - size / 2) //anchor too close to lower edge
	{
		for (int i = anchor_y - size / 2; i < img.rows; i++)
		{
			for (int j = anchor_x - size / 2; j < anchor_x + size / 2 + 1; j++)
			{
				temp.at<uchar>(i - (anchor_y - size / 2), j - (anchor_x - size / 2)) = img.at<uchar>(i, j);
			}
		}
	}
	//anchor is far enough from edge
	else if (anchor_x >= size / 2 && anchor_y >= size / 2 && anchor_x <= img.cols - 1 - size / 2 && anchor_y <= img.rows - 1 - size / 2)
	{
		for (int i = anchor_y - size / 2; i < anchor_y + size / 2; i++)
		{
			for (int j = anchor_x - size / 2; j < anchor_x + size / 2; j++)
			{
				temp.at<uchar>(i - (anchor_y - size / 2), j - (anchor_x - size / 2)) = img.at<uchar>(i, j);
			}
		}
	}
	return temp;
}

void calDisp(Mat imgL, Mat imgR, Mat& imgDisparity8U, int trim_x, int temp_size)
{
	vector<double> profile(imgR.cols - trim_x);
	for (int i = 0; i < imgL.rows; i++)
	{

		for (int j = trim_x; j < imgL.cols; j++)
		{
			Mat tempL = get_template(imgL, i, j, temp_size);
			int sum_tempL = 0;
			int	sum_sq_tempL = 0;
			for (int k = 0; k < temp_size; k++) //sum of tempL
			{
				for (int l = 0; l < temp_size; l++)
				{
					sum_tempL += tempL.at<uchar>(k, l);
				}
			}
			for (int k = 0; k < temp_size; k++) //sum of squared tempL
			{
				for (int l = 0; l < temp_size; l++)
				{
					sum_sq_tempL += pow(tempL.at<uchar>(k, l), 2);
				}
			}
			/*now we have template from left image,
			* next we will compare it to templates from right image
			* those on the same x-axis as anchor point*/
			int profile_index = 0;
			for (int n = j - 40; n < j; n++)
			{
				if (n < 0)
				{
					n = 0;
				}
				else if (n > imgR.cols)
				{
					break;
				}
				Mat tempR = get_template(imgR, i, n, temp_size);
				int sum_tempR = 0;
				int sum_sq_tempR = 0;
				int sum_sq_diff = 0;
				for (int k = 0; k < temp_size; k++) //sum of tempR
				{
					for (int l = 0; l < temp_size; l++)
					{
						sum_tempR += tempR.at<uchar>(k, l);
					}
				}
				for (int k = 0; k < temp_size; k++) //sum of squared tempR
				{
					for (int l = 0; l < temp_size; l++)
					{
						sum_sq_tempR += pow(tempR.at<uchar>(k, l), 2);
					}
				}
				/*calculation
				* referenced to CV_TM_SQDIFF_NORMED */
				for (int k = 0; k < temp_size; k++) //sum of squared tempR
				{
					for (int l = 0; l < temp_size; l++)
					{
						sum_sq_diff += pow(tempL.at<uchar>(k, l) - tempR.at<uchar>(k, l), 2);
					}
				}
				double R = sum_sq_diff / (sqrt(sum_sq_tempL) * sqrt(sum_sq_tempR));
				profile[profile_index] = R;
				profile_index++;
			}
			double min = *min_element(profile.begin(), profile.begin() + profile_index);
			int min_index = min_element(profile.begin(), profile.begin() + profile_index) - profile.begin();
			int dispval = profile_index - min_index;
			if (dispval < 0)
			{
				dispval = 0;
			}
			else if (dispval > 255)
			{
				dispval = 255;
			}
			imgDisparity8U.at<uchar>(i, j) = dispval;
			profile_index = 0;
		}
	}
}

Mat intensity_correct(Mat& imgDisparity8U, int trim_x)
{
	Mat DenseMap = Mat(imgDisparity8U.rows, imgDisparity8U.cols - trim_x, CV_8UC1);
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	for (int i = 0; i < DenseMap.rows; i++)
	{
		for (int j = 0; j < DenseMap.cols; j++)
		{
			DenseMap.at<uchar>(i, j) = imgDisparity8U.at<uchar>(i, j + trim_x);
		}
	}
	for (int i = 0; i < DenseMap.rows; i++)
	{
		for (int j = 0; j < DenseMap.cols; j++)
		{
			double pxv = double(DenseMap.at<uchar>(i, j)) / 255;
			double gamma = 0.7;
			double corrected = pow(pxv, gamma) * 255;
			DenseMap.at<uchar>(i, j) = DenseMap.at<uchar>(i, j) + 10;
			corrected = corrected * 3.5;
			if (corrected > 255)
			{
				corrected = 255;
			}
			DenseMap.at<uchar>(i, j) = corrected;
		}
	}
	return DenseMap;
}

int main()
{
	//read images
	Mat imgL = imread("view0.png", 0);
	Mat imgR = imread("view1.png", 0);
	//create the image in which we will save our disparities
	Mat imgDisparity8U = Mat(imgL.rows, imgL.cols, CV_8UC1);
	int trim_x = 24;

	//resize to speed up the iteration when testing method
	//resize(imgL, imgL, Size(imgL.cols / 2, imgL.rows / 2), INTER_LINEAR);
	//resize(imgR, imgR, Size(imgR.cols / 2, imgR.rows / 2), INTER_LINEAR);
	//blur(imgL, imgL, Size(3, 3), Point(-1, -1), 4);
	//blur(imgR, imgR, Size(3, 3), Point(-1, -1), 4);

	calDisp(imgL, imgR, imgDisparity8U, trim_x, 13);
	imshow("imgL", imgL);
	imshow("imgR", imgR);
	imshow("disparity", imgDisparity8U);
	Mat DenseMap = intensity_correct(imgDisparity8U, trim_x);
	imshow("Dense Map", DenseMap);
	imwrite("disparity.png", imgDisparity8U);
	imwrite("Dense Map.png", DenseMap);
	waitKey();
}