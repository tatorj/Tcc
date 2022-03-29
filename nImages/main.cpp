#include "iostream"
#include "img_open.cpp"
#include "Detect_Images.cpp"
#include "Compute_Images.cpp"
#include "BFM_Images.cpp"
#include "Homography_Images.cpp"
#include "Good_Matches.cpp"
#include "Good_Matches_2.cpp"
#include "Good_Matches_3.cpp"

using namespace std;

int main(int argc, char const *argv[])
{
	int n,P;

	cout << "\n Quantas imagens serão analisadas?: ";
	cin >> n;
	cout << "\n Quantos pontos ja foram inseridos no e-foto?: ";
	cin >> P;
	
	vector<Mat> images = Img_Open(n);

	vector<vector<KeyPoint>> img_Det = Img_Detect(n, images);

	vector<Mat> img_Comp = Img_Compute (n, images, img_Det);

	vector<vector<vector<KeyPoint>>> img_BFM = Img_BFM (n, img_Det, img_Comp, nn_match_ratio);

	vector<Mat> homography = Homography_func (n, matched);

	vector<vector<vector<KeyPoint>>> inliers = good_Matches (n, matched, homography, inlier_threshold);

	vector<vector<vector<KeyPoint>>> smatches = good_2_Matches (n, inliers, images);

	vector<vector<vector<KeyPoint>>> pmatches = good_3_Matches (n, inliers);

	Data_Output_Akaze (n, P, pmatches, smatches);

	return 0;
}