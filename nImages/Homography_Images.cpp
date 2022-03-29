#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>      // <- declaramos findHomography
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;

vector<Mat> Homography_func (int n, vector<vector<vector<KeyPoint>>> matched) {
	
	vector<Point2f> mpts1, mpts2;
	vector<Mat> homography;
	for (size_t i = 0; i < n-1; i++){
		
		KeyPoint::convert(matched[i][0],mpts1);
		KeyPoint::convert(matched[i][1],mpts2);
		
		homography[i] = findHomography(mpts1,mpts2,RANSAC, tsize);
			    
	}
	return homography;
}


/* Computamos a matriz homográfica usando os pontos correlatos 
    vector<Point2f> mpts1, mpts2;
    KeyPoint::convert(matched1,mpts1);
    KeyPoint::convert(matched2,mpts2);
    Mat homography = findHomography(mpts1,mpts2,RANSAC, tsize);	
*/
