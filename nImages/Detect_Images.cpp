#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp> //Class for matching keypoint descriptors

using namespace std;
using namespace cv;

vector<vector<KeyPoint>> Img_Detect (int n, vector<Mat> images) {
	
	vector<vector<KeyPoint>> img_Det;
	Ptr<AKAZE> akaze = AKAZE::create();
	
	for (size_t i = 0; i < n; i++){
		
		akaze->detect(images[i],img_Det[i]);
		
	}
	return img_Det;
}

