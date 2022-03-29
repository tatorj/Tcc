#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp> //Class for matching keypoint descriptors

using namespace std;
using namespace cv;

vector<vector<vector<KeyPoint>>> good_3_Matches (int n, vector<vector<vector<KeyPoint>>> inliers )
{
	
	vector<vector<vector<KeyPoint>>> pmatches;
	for (size_t k=0; k < inliers[k][1].size(); k++){
		for(size_t i=0; i < inliers[k+1][0].size(); i++){
			if (inliers[k][1][k].pt.x == inliers[k][0][i].pt.x && inliers[k][1][k].pt.y == inliers[k][0][i].pt.y) {
				int new_i = static_cast<int>(pmatches.size());
				pmatches[k][0].push_back(inliers[k][0][k]);
				pmatches[k][1].push_back(inliers[k][1][k]);
				pmatches[k][2].push_back(inliers[k][1][i]);				
			}
		}		
	}

	return pmatches;
}

/*
	vector<KeyPoint> pmatches1, pmatches2, pmatches3;	
	for (size_t i=0; i < inliers2.size(); i++){
		for(size_t c=0; c < inliers3.size(); c++){
			if (inliers2[i].pt.x == inliers3[c].pt.x && inliers2[i].pt.y == inliers3[c].pt.y) {
				int new_i = static_cast<int>(pmatches1.size());
				pmatches1.push_back(inliers1[i]);
				pmatches2.push_back(inliers2[i]);
				pmatches3.push_back(inliers4[c]);				
			}
		}		
	}
*/