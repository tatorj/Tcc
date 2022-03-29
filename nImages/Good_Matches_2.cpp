#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp> //Class for matching keypoint descriptors

using namespace std;
using namespace cv;

vector<vector<vector<KeyPoint>>> good_2_Matches (int n, vector<vector<vector<KeyPoint>>> inliers, vector<Mat> images) {
	
	vector<vector<vector<KeyPoint>>> smatches;

	for (int k = 0; k < n-1; ++k)
	{
		float coluna = images[k+1].cols;

		for (int i = 0; i < inliers[k][1].size; ++i)
		{
			if (k==0)
			{
				if (inliers[k][1][i].pt.x < coluna * 0.4)
				{
					smatches[k][0][i].push_back(inliers[k][0][i]);
					smatches[k][1][i].push_back(inliers[k][1][i]);
				}
			}
			else if (k==n-1)
			{
				if (inliers[k][1][i].pt.x > coluna * 0.6)
				{
					smatches[k][0][i].push_back(inliers[k][0][i]);
					smatches[k][1][i].push_back(inliers[k][1][i]);
				}
			}
			else
			{
				if (inliers[k][1][i].pt.x > coluna * 0.6 && inliers[k][1][i].pt.x < coluna * 0.8)
				{
					smatches[k][0][i].push_back(inliers[k][0][i]);
					smatches[k][1][i].push_back(inliers[k][1][i]);
				}
			}
		}

	}

	return smatches;
} 

/*



    //Separando os pontos correlatos que não apareceriam nas 3 imagens e portanto não seriam inseridos no 3_images
    
    float coluna = img2.cols;
    
    vector<KeyPoint> smatches1,smatches2,smatches3,smatches4;
    for (size_t i=0; i < inliers2.size(); i++){
		if (inliers2[i].pt.x < coluna * 0.4){
			int new_i = static_cast<int>(smatches1.size());
            smatches1.push_back(inliers1[i]);
            smatches2.push_back(inliers2[i]);	
		}		
	}
	
	
	for (size_t i=0; i < inliers3.size(); i++){
		if (inliers3[i].pt.x > coluna * 0.6){
			int new_i = static_cast<int>(smatches3.size());
            smatches3.push_back(inliers3[i]);
            smatches4.push_back(inliers4[i]);	
		}		
	}
*/