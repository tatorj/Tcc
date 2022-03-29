#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp> //Class for matching keypoint descriptors

using namespace std;
using namespace cv;

vector<vector<vector<KeyPoint>>> Img_BFM (int n, vector<vector<KeyPoint>> img_Det,vector<Mat> img_Comp, const float nn_match_ratio = 0.8f) {
	
	
	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	vector<vector<vector<KeyPoint>>> matched;
	
	for (size_t k = 0; k < n-1 ; k++){
		matcher.knnMatch(img_Comp[k], img_Comp[k+1], nn_matches, 2);
		
		for(size_t i = 0; i < nn_matches.size(); i++) {
			
			DMatch first = nn_matches[i][0];
			float dist1 = nn_matches[i][0].distance;
			float dist2 = nn_matches[i][1].distance;
			
			if(dist1 < nn_match_ratio * dist2) {
				matched[k][0][i].push_back(img_Det[k][first.queryIdx]);
				matched[k][1][i].push_back(img_Det[k+1][first.trainIdx]);
		}
	}
	return matched;
}
/*
	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);
	vector<KeyPoint> matched1, matched2;
	for(size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;
		if(dist1 < nn_match_ratio * dist2) {
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
	}
*/
