#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp> //Class for matching keypoint descriptors

using namespace std;
using namespace cv;

vector<vector<vector<KeyPoint>>>  good_Matches (int n, vector<vector<vector<KeyPoint>>> matched, vector<Mat> homography, const float inlier_threshold = 2.5f)
{
	
	vector<DMatch> good_matches;
	vector<vector<vector<KeyPoint>>> inliers;
	Mat col = Mat::ones(3, 1, CV_64F);
	
	for (size_t k = 0; k < n-1; k++){
		
		for(size_t i = 0; i < matched[k][0].size(); i++) {
			col.at<double>(0) = matched[k][0][i].pt.x;
			col.at<double>(1) = matched[k][0][i].pt.y;
			col = homography[i] * col;
			col /= col.at<double>(2);
			double dist = sqrt( pow(col.at<double>(0) - matched[k][1][i].pt.x, 2) +
								pow(col.at<double>(1) - matched[k][1][i].pt.y, 2));

			if(dist < inlier_threshold) {
				int new_i = static_cast<int>(inliers[k].size());
				inliers[k][0].push_back(matched[k][0][i]);
				inliers[k][1].push_back(matched[k][1][i]);
		}
		//                /\inliers[k][1][i]??
		
		}
	}
	return inliers;
}

/*
	vector<DMatch> good_matches;
	vector<KeyPoint> inliers1, inliers2;
	Mat col = Mat::ones(3, 1, CV_64F);
	for(size_t i = 0; i < matched1.size(); i++) {
		col.at<double>(0) = matched1[i].pt.x;
		col.at<double>(1) = matched1[i].pt.y;
		col = homography * col;
		col /= col.at<double>(2);
		double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
							pow(col.at<double>(1) - matched2[i].pt.y, 2));
		if(dist < inlier_threshold) {
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			good_matches.push_back(DMatch(new_i, new_i, 0)); // É apenas necessário para o desenho dos pontos. Que não é o foco do projeto.
		}
	}
*/
