#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp> //Class for matching keypoint descriptors

using namespace std;
using namespace cv;

vector<Mat> Img_Compute (int n, vector<Mat> images, vector<vector<KeyPoint>> img_Det) {
	
	vector<Mat> img_Comp;
    Ptr<AKAZE> akaze = AKAZE::create();
	
	for (size_t i = 0; i < n; i++){
		
		akaze->compute(images[i],img_Det[i],img_Comp[i]);
		
	    
	}
	return img_Comp;
}

