#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>      // <- declaramos findHomography
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const float inlier_threshold = 2.5f; // Distance threshold to identify inliers with homography check
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main(int argc, char* argv[]) {
	
	// Iniciação do calculo de tempo
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
	auto t1 = high_resolution_clock::now();
	
	// O parâmetro de homografia foi substituído por um limiar
    CommandLineParser parser(argc, argv,
                             "{@img1 | 16.bmp | input image 1}"
                             "{@img2 | 17.bmp | input image 2}"
                             //"{@minH | 400    | input threshold size}"
                             "{@size | 3      | input threshold size}");
    
    // Imprimimos os parâmetros fornecidos
    cout << "img1 = " << parser.get<String>("@img1") 
         << "\nimg2 = " << parser.get<String>("@img2") 
         //<< "\nminH = " << parser.get<String>("@minH")
         << "\nsize = " << parser.get<String>("@size") << "\n";
    
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
	
	cout <<"Abrindo os arquivos= "
		 << ms_double.count() << "ms\n";
    
    t1 = high_resolution_clock::now();

	// Não usaremos as imagens de exemplo do opencv
    Mat img1 = imread( parser.get<String>("@img1"), IMREAD_GRAYSCALE);
    Mat img2 = imread( parser.get<String>("@img2"), IMREAD_GRAYSCALE);
    //int minH = parser.get<int>("@minH");
    double tsize = parser.get<double>("@size");
    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }
    
    t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    ms_double = t2 - t1;

    cout <<"Lendo os arquivos= "
		 << ms_double.count() << "ms\n";
    
    // A matriz homográfica não é mais carregada de um arquivo
    //Mat homography;
    //FileStorage fs( samples::findFile( parser.get<String>("@homography") ), FileStorage::READ);
    //fs.getFirstTopLevelNode() >> homography;
    
    t1 = high_resolution_clock::now();
    
    // O código segue o mesmo do tutorial com AKAZE
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    Ptr<SIFT> SIFT = SIFT::create();
    SIFT->detectAndCompute(img1, noArray(), kpts1, desc1);
    SIFT->detectAndCompute(img2, noArray(), kpts2, desc2);
    BFMatcher matcher(NORM_L2);
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
    
    t2 = high_resolution_clock::now();
    
    /* Getting number of milliseconds as a double. */
    ms_double = t2 - t1;

    cout <<"Detect and Compute= "
		 << ms_double.count() << "ms\n";
    
    t1 = high_resolution_clock::now();
    
    // Computamos a matriz homográfica usando os pontos correlatos 
    vector<Point2f> mpts1, mpts2;
    KeyPoint::convert(matched1,mpts1);
    KeyPoint::convert(matched2,mpts2);
    Mat homography = findHomography(mpts1,mpts2,RANSAC, tsize);	
    
    t2 = high_resolution_clock::now();
 
    /* Getting number of milliseconds as a double. */
    ms_double = t2 - t1;

    cout <<"Homography= "
		 << ms_double.count() << "ms\n";
    
    t1 = high_resolution_clock::now();
    
    // O código segue o mesmo do tutorial com AKAZE
    vector<DMatch> good_matches;
    vector<KeyPoint> inliers1, inliers2;
    for(size_t i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
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
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    
    t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    ms_double = t2 - t1;

    cout <<"Correlatos= "
		 << ms_double.count() << "ms\n";
    
    t1 = high_resolution_clock::now();
    
    Mat res;
    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    imwrite("SIFT_result.png", res);
    
    t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    ms_double = t2 - t1;

    cout <<"Draw matches= "
		 << ms_double.count() << "ms\n";
    
    double inlier_ratio = inliers1.size() / (double) matched1.size();
    cout << "SIFT Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl;
    cout << "# Inliers:                            \t" << inliers1.size() << endl;
    cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
    cout << endl;
    //imshow("result", res);
    waitKey();
    return 0;
}
