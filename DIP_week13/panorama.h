#pragma once
#include <iostream>
#include "opencv2/core/core.hpp" //Mat class ��
#include "opencv2/highgui/highgui.hpp" //imshow�� gui�� ���õ� �Լ����� ���Ե� ���
#include "opencv2/imgproc/imgproc.hpp" //�̹��� ó��

#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>

#include "opencv2/stitching.hpp"
#include <opencv2/calib3d.hpp>

using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;

void ex_panorama_simple();
void ex_panorama();
Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches);
void BooknScene();